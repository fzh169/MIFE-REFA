import matplotlib.pyplot as plt
import torch
import sys
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange, repeat


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args

        self.fe = FeatureExtractor(dim=(32, 64, 128, 256))

        self.flow0 = BasicFlow(dim=64,  up_factor=2)
        self.flow1 = BasicFlow(dim=128, up_factor=4)
        self.flow2 = BasicFlow(dim=256, up_factor=8)

        self.fn = FusionNet(dim=(16, 32, 64, 96), in_ch=6, out_ch=2)
        self.om = OcclusionMask(dim=32, in_ch=3)

        self.ce = ContextExtractor(dim=(16, 32, 64, 96))
        self.sy = SynthesisNet(dim=((16, 32, 64, 96), (96, 192, 288)), in_ch=17, out_ch=3)

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        x = 8
        if h0 % x != 0:
            pad_h = x - (h0 % x)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h])
            frame2 = F.pad(frame2, [0, 0, 0, pad_h])

        if w0 % x != 0:
            pad_w = x - (w0 % x)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0])
            frame2 = F.pad(frame2, [0, pad_w, 0, 0])

        mean_f = torch.cat([frame0, frame2], dim=1).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

        feat00, feat10, feat20 = self.fe(frame0 - mean_f)
        feat02, feat12, feat22 = self.fe(frame2 - mean_f)

        flow10_0, flow12_0, alpha_0 = self.flow0(feat00, feat02)
        flow10_1, flow12_1, alpha_1 = self.flow1(feat10, feat12)
        flow10_2, flow12_2, alpha_2 = self.flow2(feat20, feat22)

        flow10 = self.fn(torch.cat([flow10_0, flow10_1, flow10_2], dim=1))
        flow12 = self.fn(torch.cat([flow12_0, flow12_1, flow12_2], dim=1))

        f10 = self.backwarp(flow10, frame0)
        f12 = self.backwarp(flow12, frame2)

        alpha = self.om(torch.cat([alpha_0, alpha_1, alpha_2], dim=1))
        frame1_hat = alpha * f10 + (1 - alpha) * f12

        c0 = self.ce(frame0, flow10)
        c1 = self.ce(frame2, flow12)
        res = self.sy(torch.cat([frame0, frame2, f10, f12, flow10, flow12, alpha], dim=1), c0, c1) * 2 - 1
        frame1_hat = torch.clamp(frame1_hat + res, 0, 1)[:, :, 0:h0, 0:w0]

        if self.training:

            g0 = self.generate(flow10_0, flow12_0, alpha_0, frame0, frame2)[:, :, 0:h0, 0:w0]
            g1 = self.generate(flow10_1, flow12_1, alpha_1, frame0, frame2)[:, :, 0:h0, 0:w0]
            g2 = self.generate(flow10_2, flow12_2, alpha_2, frame0, frame2)[:, :, 0:h0, 0:w0]

            return {'frame1': frame1_hat, 'g0': g0, 'g1': g1, 'g2': g2}
        else:

            return frame1_hat

    def generate(self, flow10, flow12, alpha, frame0, frame2):

        f10 = self.backwarp(flow10, frame0)
        f12 = self.backwarp(flow12, frame2)

        frame1_hat = alpha * f10 + (1 - alpha) * f12

        return frame1_hat

    def backwarp(self, flow, frame):

        B, _, H, W = flow.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)

        coords = base + flow

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        frame_hat = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return frame_hat


class BasicFlow(nn.Module):
    def __init__(self, dim, up_factor):
        super(BasicFlow, self).__init__()

        self.up_factor = up_factor

        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1),
            ResBlock(dim, dim), ResBlock(dim, dim), ResBlock(dim, dim),
            nn.Conv2d(dim, 5, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, feat0, feat2):

        out = self.conv(torch.cat([feat0, feat2], dim=1))

        flow10 = out[:, 0:2]
        flow12 = out[:, 2:4]
        alpha = torch.sigmoid(out[:, 4:5])

        flow10 = F.interpolate(flow10, scale_factor=self.up_factor, mode="bilinear") * self.up_factor
        flow12 = F.interpolate(flow12, scale_factor=self.up_factor, mode="bilinear") * self.up_factor
        alpha = F.interpolate(alpha, scale_factor=self.up_factor, mode="bilinear")

        return flow10, flow12, alpha


class FeatureExtractor(nn.Module):
    def __init__(self, dim):
        super(FeatureExtractor, self).__init__()

        self.emb = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)
        self.fe0 = FEBlock(dim[0], dim[1])
        self.fe1 = FEBlock(dim[1], dim[2])
        self.fe2 = FEBlock(dim[2], dim[3])

    def forward(self, frame):
        feat0 = self.fe0(self.emb(frame))
        feat1 = self.fe1(feat0)
        feat2 = self.fe2(feat1)

        return feat0, feat1, feat2


class FusionNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super(FusionNet, self).__init__()

        self.init = ResBlock(in_ch, dim[0])

        self.fd0 = DownBlock(dim[0], dim[1])
        self.fd1 = DownBlock(dim[1], dim[2])
        self.fd2 = DownBlock(dim[2], dim[3])

        self.fu2 = UpBlock(dim[3], dim[2])
        self.fu1 = UpBlock(dim[2], dim[1])
        self.fu0 = UpBlock(dim[1], dim[0])

        self.last = nn.Conv2d(in_channels=dim[0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, flow):
        return self.fnet(flow)

    def fnet(self, flow):
        feat00 = self.init(flow)

        feat01 = self.fd0(feat00)
        feat02 = self.fd1(feat01)
        feat03 = self.fd2(feat02)

        feat12 = self.fu2(feat03, feat02)
        feat11 = self.fu1(feat12, feat01)
        feat10 = self.fu0(feat11, feat00)

        return self.last(feat10)


class OcclusionMask(nn.Module):
    def __init__(self, dim, in_ch):
        super(OcclusionMask, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv2(self.relu1(self.conv1(x)))
        out = torch.sigmoid(x1 + x2)

        return out


class ContextExtractor(nn.Module):
    def __init__(self, dim):
        super(ContextExtractor, self).__init__()

        self.emb = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)
        self.ce0 = CEBlock(dim[0], dim[1])
        self.ce1 = CEBlock(dim[1], dim[2])
        self.ce2 = CEBlock(dim[2], dim[3])

    def forward(self, frame, flow):
        feat0 = self.ce0(self.emb(frame))
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f0 = self.backwarp(flow, feat0)

        feat1 = self.ce1(feat0)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f1 = self.backwarp(flow, feat1)

        feat2 = self.ce2(feat1)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f2 = self.backwarp(flow, feat2)

        return [f0, f1, f2]

    def backwarp(self, flow, feat):
        B, _, H, W = flow.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)

        coords = base + flow

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return feat


class SynthesisNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super(SynthesisNet, self).__init__()

        self.init = ResBlock(in_ch, dim[0][0])

        self.sd0 = DownBlock(dim[0][0], dim[0][1])
        self.sd1 = DownBlock(dim[1][0], dim[0][2])
        self.sd2 = DownBlock(dim[1][1], dim[0][3])

        self.su2 = UpBlock(dim[1][2], dim[0][2])
        self.su1 = UpBlock(dim[0][2], dim[0][1])
        self.su0 = UpBlock(dim[0][1], dim[0][0])

        self.last = nn.Conv2d(in_channels=dim[0][0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, feat, c0, c1):
        feat00 = self.init(feat)

        feat01 = self.sd0(feat00)
        feat02 = self.sd1(torch.cat([feat01, c0[0], c1[0]], dim=1))
        feat03 = self.sd2(torch.cat([feat02, c0[1], c1[1]], dim=1))

        feat12 = self.su2(torch.cat([feat03, c0[2], c1[2]], dim=1), feat02)
        feat11 = self.su1(feat12, feat01)
        feat10 = self.su0(feat11, feat00)

        return torch.sigmoid(self.last(feat10))


class FEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FEBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = nn.Sequential(ResBlock(out_ch, out_ch), ResBlock(out_ch, out_ch))
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class CEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CEBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)

        return feat


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

        if norm:
            self.norm = nn.LayerNorm(out_ch)
        else:
            self.norm = lambda x: x

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(UpBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

        if norm:
            self.norm = nn.LayerNorm(out_ch)
        else:
            self.norm = lambda x: x

    def forward(self, feat1, feat0):
        feat = self.up(feat1) + feat0
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        # self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        # self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.PReLU()

        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1))
        else:
            self.downsample = lambda x: x

    def forward(self, x):

        out = self.conv1(x)
        # out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        # out = self.norm2(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu2(out)

        return out

