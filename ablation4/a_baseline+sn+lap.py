import matplotlib.pyplot as plt
import torch
import sys
import cv2
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange, repeat


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args

        self.fe = FeatureExtractor(dim=(32, 64, 128, 192, 256))
        self.ee = EdgeExtractor(dim=(32, 64, 128, 192, 256))
        self.sn = ScoreNet(dim=(16, 32, 64, 96), in_ch=27, out_ch=9, kernel=3, dilation=(8, 16, 32, 64))

        self.flow0 = RefineFlow(dim=64,  part_size=8,  up_factor=8,  kernel=3)
        self.flow1 = RefineFlow(dim=128, part_size=8,  up_factor=16,  kernel=3)
        self.flow2 = RefineFlow(dim=192, part_size=16, up_factor=32, kernel=3)
        self.flow3 = RefineFlow(dim=256, part_size=16, up_factor=64, kernel=3)

        self.rm = RangeMask(dim=32)
        self.fn = FusionNet(dim=(16, 32, 64, 96), in_ch=2, out_ch=2)
        self.om = OcclusionMask(dim=32, in_ch=2)

        self.ce = ContextExtractor(dim=(16, 32, 64, 96))
        self.sy = SynthesisNet(dim=((16, 32, 64, 96), (96, 192, 288)), in_ch=17, out_ch=3)

        self.cn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        x = 64
        if h0 % x != 0:
            pad_h = x - (h0 % x)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h])
            frame2 = F.pad(frame2, [0, 0, 0, pad_h])

        if w0 % x != 0:
            pad_w = x - (w0 % x)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0])
            frame2 = F.pad(frame2, [0, pad_w, 0, 0])

        mean_f = torch.cat([frame0, frame2], dim=1).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

        frames0 = F.interpolate(frame0, scale_factor=0.25, mode="bilinear")
        frames2 = F.interpolate(frame2, scale_factor=0.25, mode="bilinear")

        feat00, feat10, feat20, feat30 = self.fe(frames0 - mean_f)
        feat02, feat12, feat22, feat32 = self.fe(frames2 - mean_f)

        lap00, lap10, lap20, lap30 = self.ee(frames0)
        lap02, lap12, lap22, lap32 = self.ee(frames2)

        score00, score10, score20, score30 = self.sn(frame0 - mean_f)
        score02, score12, score22, score32 = self.sn(frame2 - mean_f)

        flow_0, range_0, corr02_0, corr20_0 = self.flow0(feat00, feat02, lap00, lap02, score00, score02)
        flow_1, range_1, corr02_1, corr20_1 = self.flow1(feat10, feat12, lap10, lap12, score10, score12)
        flow_2, range_2, corr02_2, corr20_2 = self.flow2(feat20, feat22, lap20, lap22, score20, score22)
        flow_3, range_3, corr02_3, corr20_3 = self.flow3(feat30, feat32, lap30, lap32, score30, score32)

        flow10 = self.fn(self.rm(flow_0[0], flow_1[0], flow_2[0], flow_3[0], range_0[0], range_1[0], range_2[0]))
        flow12 = self.fn(self.rm(flow_0[1], flow_1[1], flow_2[1], flow_3[1], range_0[1], range_1[1], range_2[1]))

        flow02, corr02 = self.rm(flow_0[2], flow_1[2], flow_2[2], flow_3[2], range_0[2], range_1[2], range_2[2],
                                 corr02_0, corr02_1, corr02_2, corr02_3)
        flow20, corr20 = self.rm(flow_0[3], flow_1[3], flow_2[3], flow_3[3], range_0[3], range_1[3], range_2[3],
                                 corr20_0, corr20_1, corr20_2, corr20_3)

        flow02, corr02 = self.fn(flow02), self.cn(corr02)
        flow20, corr20 = self.fn(flow20), self.cn(corr20)

        f10 = self.backwarp(flow10, frame0)
        f12 = self.backwarp(flow12, frame2)

        rad02 = (flow02[:, 0] ** 2 + flow02[:, 1] ** 2 + 1e-6) ** 0.5
        rad20 = (flow20[:, 0] ** 2 + flow20[:, 1] ** 2 + 1e-6) ** 0.5

        tmp02 = torch.cat([rad02.unsqueeze(1), corr02], dim=1)
        tmp20 = torch.cat([rad20.unsqueeze(1), corr20], dim=1)

        alpha = self.om(tmp20 - tmp02)
        frame1_hat = alpha * f10 + (1 - alpha) * f12

        c0 = self.ce(frame0, flow10)
        c1 = self.ce(frame2, flow12)
        res = self.sy(torch.cat([frame0, frame2, f10, f12, flow10, flow12, alpha], dim=1), c0, c1) * 2 - 1
        frame1_hat = torch.clamp(frame1_hat + res, 0, 1)[:, :, 0:h0, 0:w0]

        if self.training:

            g0 = self.generate(flow_0, corr02_0, corr20_0, frame0, frame2)[:, :, 0:h0, 0:w0]
            g1 = self.generate(flow_1, corr02_1, corr20_1, frame0, frame2)[:, :, 0:h0, 0:w0]
            g2 = self.generate(flow_2, corr02_2, corr20_2, frame0, frame2)[:, :, 0:h0, 0:w0]
            g3 = self.generate(flow_3, corr02_3, corr20_3, frame0, frame2)[:, :, 0:h0, 0:w0]

            return {'frame1': frame1_hat, 'g0': g0, 'g1': g1, 'g2': g2, 'g3': g3}
        else:

            return frame1_hat

    def generate(self, flow, corr02, corr20, frame0, frame2):

        f10 = self.backwarp(flow[0], frame0)
        f12 = self.backwarp(flow[1], frame2)

        rad02 = (flow[2][:, 0] ** 2 + flow[2][:, 1] ** 2 + 1e-6) ** 0.5
        rad20 = (flow[3][:, 0] ** 2 + flow[3][:, 1] ** 2 + 1e-6) ** 0.5

        tmp02 = torch.cat([rad02.unsqueeze(1), corr02], dim=1)
        tmp20 = torch.cat([rad20.unsqueeze(1), corr20], dim=1)

        alpha = self.om(tmp20 - tmp02)
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


class RefineFlow(nn.Module):
    def __init__(self, dim, part_size, up_factor, kernel):
        super(RefineFlow, self).__init__()

        self.scale = dim ** -0.5
        self.part_h = part_size
        self.part_w = part_size
        self.up_factor = up_factor

        self.proj_q0 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.proj_k0 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.proj_q1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.proj_k1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        self.bias_size = part_size
        self.bias_table = nn.Parameter(torch.zeros((2 * self.bias_size - 1) ** 2, 1))

        coords_h = torch.arange(self.bias_size)
        coords_w = torch.arange(self.bias_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2 h w
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (hw) (hw) 2
        relative_coords[:, :, 0] += self.bias_size - 1
        relative_coords[:, :, 1] += self.bias_size - 1
        relative_coords[:, :, 0] *= 2 * self.bias_size - 1
        self.bias_index = relative_coords.sum(-1).view(self.bias_size, self.bias_size, self.bias_size, self.bias_size)

        self.kernel_size = kernel
        self.padding = int(self.kernel_size / 2) * self.up_factor

    def forward(self, feat0, feat2, lap0, lap2, score0, score2):

        B, C, H, W = feat0.shape

        pad_h, pad_w = 0, 0
        if H % self.part_h != 0:
            pad_h = self.part_h - (H % self.part_h)
            feat0 = F.pad(feat0, [0, 0, 0, pad_h])
            feat2 = F.pad(feat2, [0, 0, 0, pad_h])
            lap0 = F.pad(lap0, [0, 0, 0, pad_h])
            lap2 = F.pad(lap2, [0, 0, 0, pad_h])

        if W % self.part_w != 0:
            pad_w = self.part_w - (W % self.part_w)
            feat0 = F.pad(feat0, [0, pad_w, 0, 0])
            feat2 = F.pad(feat2, [0, pad_w, 0, 0])
            lap0 = F.pad(lap0, [0, pad_w, 0, 0])
            lap2 = F.pad(lap2, [0, pad_w, 0, 0])

        s1 = int((H + pad_h) / self.part_h)
        s2 = int((W + pad_w) / self.part_w)

        q00, q02, q10, q12 = self.proj_q0(feat0), self.proj_q0(feat2), self.proj_q1(lap0), self.proj_q1(lap2)
        k00, k02, k10, k12 = self.proj_k0(feat0), self.proj_k0(feat2), self.proj_k1(lap0), self.proj_k1(lap2)

        dx = torch.arange(0.0, self.part_w, device="cuda")
        dy = torch.arange(0.0, self.part_h, device="cuda")
        meshy, meshx = torch.meshgrid(dy, dx)
        pos = torch.stack((meshx, meshy), dim=0).unsqueeze(0)
        pos = rearrange(pos, 'b f h w -> b f (h w)')

        index = self.bias_index.reshape(-1)
        bias = self.bias_table[index].reshape(self.part_h * self.part_w, self.part_h * self.part_w, -1).permute(2, 0, 1)

        flow12, corr12, range1, flow02, corr02, range0 = self.estimate((q00, q10), (k02, k12), pos, bias, s1, s2, H, W)
        flow10, corr10, range1, flow20, corr20, range0 = self.estimate((q02, q12), (k00, k10), pos, bias, s1, s2, H, W)

        f1 = int(self.up_factor / 2)
        f0 = self.up_factor

        flow10, corr10, range10 = self.refine(flow10, corr10, range1, score0, f1, False)
        flow12, corr12, range12 = self.refine(flow12, corr12, range1, score2, f1, False)

        flow02, corr02, range02 = self.refine(flow02, corr02, range0, score0, f0, True)
        flow20, corr20, range20 = self.refine(flow20, corr20, range0, score2, f0, True)

        return (flow10, flow12, flow02, flow20), (range10, range12, range02, range20), corr02, corr20

    def refine(self, flow, corr, rang, score, factor, target):

        # flow = repeat(flow, 'b f h w -> b f (h n1) (w n2)', n1=factor, n2=factor) * factor
        # corr = repeat(corr, 'b n h w -> b n (h n1) (w n2)', n1=factor, n2=factor)
        # rang = repeat(rang, 'b f h w -> b f (h n1) (w n2)', n1=factor, n2=factor) * factor

        flow = F.interpolate(flow, scale_factor=factor, mode="bilinear") * factor
        corr = F.interpolate(corr, scale_factor=factor, mode="bilinear")
        rang = F.interpolate(rang, scale_factor=factor, mode="bilinear") * factor

        B, _, H, W = flow.shape

        flow = F.unfold(flow, kernel_size=self.kernel_size, dilation=self.up_factor, padding=self.padding)
        corr = F.unfold(corr, kernel_size=self.kernel_size, dilation=self.up_factor, padding=self.padding)
        rang = F.unfold(rang, kernel_size=self.kernel_size, dilation=self.up_factor, padding=self.padding)

        flow = rearrange(flow, 'b (f k) (h w) -> b f k h w', f=2, h=H)
        corr = rearrange(corr, 'b k (h w) -> b k h w', h=H)
        rang = rearrange(rang, 'b (f k) (h w) -> b f k h w', f=4, h=H)

        if not target:
            score = self.sample(flow, score)

        smax = torch.softmax(score * corr, dim=1)
        corr = torch.sum(corr * smax, dim=1).unsqueeze(1)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=2)
        rang = torch.sum(rang * smax.unsqueeze(1), dim=2)

        return flow, corr, rang

    def sample(self, flow, score):
        B, _, H, W = score.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0)

        coords = base.unsqueeze(0).unsqueeze(2) + flow
        coords = rearrange(coords, 'b f k h w -> (b k) f h w')

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        score = rearrange(score, 'b k h w -> (b k) h w').unsqueeze(1)
        score = F.grid_sample(score, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(1)
        score = rearrange(score, '(b k) h w -> b k h w', k=self.kernel_size ** 2)

        return score

    def estimate(self, q, k, pos, bias, s1, s2, H, W):

        flow1_00, corr1_00, range1_00, flow0_00, corr0_00, range0_00 = self.part(q, k, pos, bias, (False, False))
        flow1_01, corr1_01, range1_01, flow0_01, corr0_01, range0_01 = self.part(q, k, pos, bias, (False, True))
        flow1_10, corr1_10, range1_10, flow0_10, corr0_10, range0_10 = self.part(q, k, pos, bias, (True, False))
        flow1_11, corr1_11, range1_11, flow0_11, corr0_11, range0_11 = self.part(q, k, pos, bias, (True, True))

        flow1 = self.splice(flow1_00, flow1_01, flow1_10, flow1_11, 2, s1, s2, H, W)
        corr1 = self.splice(corr1_00, corr1_01, corr1_10, corr1_11, 2, s1, s2, H, W)
        flow0 = self.splice(flow0_00, flow0_01, flow0_10, flow0_11, 1, s1, s2, H, W)
        corr0 = self.splice(corr0_00, corr0_01, corr0_10, corr0_11, 1, s1, s2, H, W)

        range0 = self.splice(range0_00, range0_01, range0_10, range0_11, 1, s1, s2, H, W)
        range1 = self.splice(range1_00, range1_01, range1_10, range1_11, 2, s1, s2, H, W)

        return flow1, corr1, range1, flow0, corr0, range0

    def part(self, q, k, pos, bias, shift):

        q0, q1, k0, k1 = q[0], q[1], k[0], k[1]

        B, C, H, W = q0.shape

        shift_h, shift_w = 0, 0
        if shift[0]:
            shift_h = int(self.part_h / 2)
        if shift[1]:
            shift_w = int(self.part_w / 2)

        mask = None
        if shift[0] or shift[1]:
            q0 = torch.roll(q0, shifts=(-shift_h, -shift_w), dims=(2, 3))
            k0 = torch.roll(k0, shifts=(-shift_h, -shift_w), dims=(2, 3))
            q1 = torch.roll(q1, shifts=(-shift_h, -shift_w), dims=(2, 3))
            k1 = torch.roll(k1, shifts=(-shift_h, -shift_w), dims=(2, 3))

            mask = torch.zeros((H, W)).type_as(q0)
            mask = self.mask(mask, shift_h, shift_w)
            mask = repeat(mask, 's t1 t2 -> (b s) t1 t2', b=B)

        q0 = rearrange(q0, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)
        k0 = rearrange(k0, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)
        q1 = rearrange(q1, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)
        k1 = rearrange(k1, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)

        corr = (torch.einsum('blk,btk->blt', q0, k0) + torch.einsum('blk,btk->blt', q1, k1)) * self.scale + bias

        if mask is not None:
            corr = corr + mask

        flow0, corr0, range0 = self.flow_bsd(corr, pos)
        flow1, corr1, range1 = self.flow_mid(corr, pos)

        return flow1, corr1, range1, flow0, corr0, range0

    def mask(self, mask, h, w):

        if h != 0:
            h_slices = (slice(0, -h * 2), slice(-h * 2, -h), slice(-h, None))
        else:
            h_slices = (slice(0, None),)
        if w != 0:
            w_slices = (slice(0, -w * 2), slice(-w * 2, -w), slice(-w, None))
        else:
            w_slices = (slice(0, None),)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[h, w] = cnt
                cnt += 1
        mask_windows = rearrange(mask, '(s1 h) (s2 w) -> (s1 s2) (h w)', h=self.part_h, w=self.part_w)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (s1 s2) (hw) (hw)
        mask = mask.masked_fill(mask != 0, float(-10000.0)).masked_fill(mask == 0, float(0.0))

        return mask

    def flow_bsd(self, corr, pos):

        cut = (slice(int(self.part_h / 4), int(-self.part_h / 4)),
               slice(int(self.part_w / 4), int(-self.part_w / 4)))

        corr = rearrange(corr, 'b (h w) t -> b h w t', h=self.part_h)[:, cut[0], cut[1], :]
        corr = rearrange(corr, 'b h w t -> b (h w) t')

        bx = torch.arange(0.0, self.part_w, device="cuda")
        by = torch.arange(0.0, self.part_h, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)[:, :, cut[0], cut[1]]

        bh = base.size()[2]
        base = rearrange(base, 'b f h w -> b f (h w)')
        flow = pos.unsqueeze(2) - base.unsqueeze(3)

        maxx = torch.max(flow.clone().detach(), dim=3)[0]
        minn = torch.min(flow.clone().detach(), dim=3)[0]
        rang = torch.cat([minn, maxx], dim=1)
        rang = rearrange(rang, 'b f (h w) -> b f h w', h=bh)

        smax = torch.softmax(corr, dim=2)
        corr = torch.sum(corr * smax, dim=2)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=3)

        corr = rearrange(corr, 'b (h w) -> b h w', h=bh).unsqueeze(1)
        flow = rearrange(flow, 'b f (h w) -> b f h w', h=bh)
        rang = rang.repeat(flow.size()[0], 1, 1, 1)

        return flow, corr, rang

    def flow_mid(self, corr, pos):

        corr, base = self.transform(corr)

        bh = base.size()[2]
        base = rearrange(base, 'b f h w -> b f (h w)')
        flow = pos.unsqueeze(2) - base.unsqueeze(3)

        maxx = torch.max(flow.clone().detach(), dim=3)[0]
        minn = torch.min(flow.clone().detach(), dim=3)[0]
        rang = torch.min(torch.stack([maxx, -minn], dim=3), dim=3)[0]

        smax = torch.softmax(corr, dim=2)
        corr = torch.sum(corr * smax, dim=2)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=3)

        corr = rearrange(corr, 'b (h w)-> b h w', h=bh).unsqueeze(1)
        flow = rearrange(flow, 'b f (h w) -> b f h w', h=bh)
        rang = rearrange(rang, 'b f (h w) -> b f h w', h=bh)
        bh = bh - 1

        corr = torch.cat([corr[:, :, :-1, :-1], corr[:, :, :-1, 1:], corr[:, :, 1:, :-1], corr[:, :, 1:, 1:]], dim=1)
        flow = torch.cat([flow[:, :, :-1, :-1], flow[:, :, :-1, 1:], flow[:, :, 1:, :-1], flow[:, :, 1:, 1:]], dim=1)
        rang = torch.cat([rang[:, :, :-1, :-1], rang[:, :, :-1, 1:], rang[:, :, 1:, :-1], rang[:, :, 1:, 1:]], dim=1)

        corr = rearrange(corr, 'b n h w -> b (h w) n')
        flow = rearrange(flow, 'b (n f) h w -> b f (h w) n', f=2) * 2
        rang = rearrange(rang, 'b (n f) h w -> b f (h w) n', f=2) * 2

        smax = torch.softmax(corr, dim=2)
        corr = torch.sum(corr * smax, dim=2)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=3)
        rang = torch.sum(rang * smax.unsqueeze(1), dim=3)
        rang = torch.cat([-rang, rang], dim=1)

        corr = rearrange(corr, 'b (h w) -> b h w', h=bh).unsqueeze(1)
        flow = rearrange(flow, 'b f (h w) -> b f h w', h=bh)
        rang = rearrange(rang, 'b f (h w) -> b f h w', h=bh)

        return flow, corr, rang

    def transform(self, corr):

        ch, cw = int(self.part_h / 2) - 1, int(self.part_w / 2) - 1
        cut_h, cut_w = (slice(ch, -ch), slice(cw, -cw))

        corr = rearrange(corr, 'b (h0 w0) (h2 w2) -> h2 w2 b h0 w0', h0=self.part_h, h2=self.part_h)

        temp = []
        for i in range(0, self.part_h):
            s = F.pad(corr[i], [0, 0, i, self.part_h - i - 1])[:, :, cut_h, :]
            temp.append(s.unsqueeze(0))
        corr = torch.cat(temp, dim=0)

        temp = []
        for i in range(0, self.part_w):
            s = F.pad(corr[:, i], [i, self.part_w - i - 1, 0, 0])[:, :, :, cut_w]
            temp.append(s.unsqueeze(1))
        corr = torch.cat(temp, dim=1)

        corr = rearrange(corr, 'h2 w2 b h1 w1 -> b (h1 w1) (h2 w2)')

        bx = torch.arange(0.0, self.part_w - 0.5, step=0.5, device="cuda")
        by = torch.arange(0.0, self.part_h - 0.5, step=0.5, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)[:, :, cut_h, cut_w]

        return corr, base

    def splice(self, feat00, feat01, feat10, feat11, factor, s1, s2, H, W):

        feat0 = torch.cat([feat00, feat01], dim=3)
        feat1 = torch.cat([feat10, feat11], dim=3)
        feat = torch.cat([feat0, feat1], dim=2)
        feat = rearrange(feat, '(b s1 s2) k h w -> b k (s1 h) (s2 w)', s1=s1, s2=s2)

        shift_h = int(self.part_h / 4 * factor)
        shift_w = int(self.part_w / 4 * factor)

        feat = torch.roll(feat, shifts=(shift_h, shift_w), dims=(2, 3))
        feat = feat[:, :, 0:H * factor, 0:W * factor]

        return feat


class FeatureExtractor(nn.Module):
    def __init__(self, dim):
        super(FeatureExtractor, self).__init__()

        self.emb = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)
        self.fe0 = FEBlock(dim[0], dim[1])
        self.fe1 = FEBlock(dim[1], dim[2])
        self.fe2 = FEBlock(dim[2], dim[3])
        self.fe3 = FEBlock(dim[3], dim[4])

    def forward(self, frame):

        feat0 = self.fe0(self.emb(frame))
        feat1 = self.fe1(feat0)
        feat2 = self.fe2(feat1)
        feat3 = self.fe3(feat2)

        return feat0, feat1, feat2, feat3


class EdgeExtractor(nn.Module):
    def __init__(self, dim):
        super(EdgeExtractor, self).__init__()

        self.init = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)

        self.ed0 = EDBlock(dim[0], dim[1])
        self.ed1 = EDBlock(dim[1], dim[2])
        self.ed2 = EDBlock(dim[2], dim[3])
        self.ed3 = EDBlock(dim[3], dim[4])

        self.eu3 = EUBlock(dim[4], dim[3])
        self.eu2 = EUBlock(dim[3], dim[2])
        self.eu1 = EUBlock(dim[2], dim[1])

    def forward(self, frame):

        feat00 = self.init(self.laplacian(frame))

        feat01 = self.ed0(feat00)
        feat02 = self.ed1(feat01)
        feat03 = self.ed2(feat02)
        feat04 = self.ed3(feat03)

        feat14 = feat04
        feat13 = self.eu3(feat14, feat03)
        feat12 = self.eu2(feat13, feat02)
        feat11 = self.eu1(feat12, feat01)

        return feat11, feat12, feat13, feat14

    def laplacian(self, frame):

        fn = frame.mul(255).byte()
        fn = fn.cpu().numpy().transpose((0, 2, 3, 1))

        fls = []
        for temp in fn:
            fb = cv2.bilateralFilter(temp, 10, 20, 20)
            fl = cv2.Laplacian(fb, ddepth=cv2.CV_32F, ksize=5)
            fl = cv2.convertScaleAbs(fl)
            fl = torch.from_numpy(fl.transpose((2, 0, 1))).unsqueeze(0).cuda()
            fl = fl.float().div(255)
            fls.append(fl)
        fls = torch.cat(fls, dim=0)

        return fls


class ScoreNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch, kernel, dilation):
        super(ScoreNet, self).__init__()

        self.kernel_size = kernel
        self.dilation = dilation
        self.padding = int(self.kernel_size / 2) * self.dilation

        self.init = ResBlock(in_ch, dim[0])

        self.sd0 = DownBlock(dim[0], dim[1], True)
        self.sd1 = DownBlock(dim[1], dim[2], True)
        self.sd2 = DownBlock(dim[2], dim[3], True)

        self.su2 = UpBlock(dim[3], dim[2], True)
        self.su1 = UpBlock(dim[2], dim[1], True)
        self.su0 = UpBlock(dim[1], dim[0], True)

        self.last = nn.Conv2d(in_channels=dim[0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, frame):

        B, _, H, W = frame.shape

        feat0 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[0], padding=self.padding[0])
        feat1 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[1], padding=self.padding[1])
        feat2 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[2], padding=self.padding[2])
        feat3 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[3], padding=self.padding[3])

        feat0 = rearrange(feat0, 'b k (h w) -> b k h w', h=H)
        feat1 = rearrange(feat1, 'b k (h w) -> b k h w', h=H)
        feat2 = rearrange(feat2, 'b k (h w) -> b k h w', h=H)
        feat3 = rearrange(feat3, 'b k (h w) -> b k h w', h=H)

        score0 = torch.sigmoid(self.snet(feat0))
        score1 = torch.sigmoid(self.snet(feat1))
        score2 = torch.sigmoid(self.snet(feat2))
        score3 = torch.sigmoid(self.snet(feat3))

        return score0, score1, score2, score3

    def snet(self, frame):
        feat00 = self.init(frame)

        feat01 = self.sd0(feat00)
        feat02 = self.sd1(feat01)
        feat03 = self.sd2(feat02)

        feat12 = self.su2(feat03, feat02)
        feat11 = self.su1(feat12, feat01)
        feat10 = self.su0(feat11, feat00)

        return self.last(feat10)


class RangeMask(nn.Module):
    def __init__(self, dim):
        super(RangeMask, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, flow0, flow1, flow2, flow3, range0, range1, range2, corr0=None, corr1=None, corr2=None, corr3=None):

        range0 = rearrange(range0, 'b (m f) h w -> b m f h w', f=2)
        range1 = rearrange(range1, 'b (m f) h w -> b m f h w', f=2)
        range2 = rearrange(range2, 'b (m f) h w -> b m f h w', f=2)

        mask3 = torch.cat([range2[:, 0] - flow3, flow3 - range2[:, 1]], dim=1).detach()
        mask3 = self.conv(torch.max(mask3, dim=1)[0].unsqueeze(1))
        flow2 = flow3 * mask3 + flow2 * (1 - mask3)

        mask2 = torch.cat([range1[:, 0] - flow2, flow2 - range1[:, 1]], dim=1).detach()
        mask2 = self.conv(torch.max(mask2, dim=1)[0].unsqueeze(1))
        flow1 = flow2 * mask2 + flow1 * (1 - mask2)

        mask1 = torch.cat([range0[:, 0] - flow1, flow1 - range0[:, 1]], dim=1).detach()
        mask1 = self.conv(torch.max(mask1, dim=1)[0].unsqueeze(1))
        flow0 = flow1 * mask1 + flow0 * (1 - mask1)

        # a1 = mask1.squeeze(0).squeeze(0).cpu().numpy()
        # plt.imshow(a1)
        # plt.show()
        #
        # a2 = mask2.squeeze(0).squeeze(0).cpu().numpy()
        # plt.imshow(a2)
        # plt.show()

        if corr0 is not None:
            corr2 = corr3 * mask3 + corr2 * (1 - mask3)
            corr1 = corr2 * mask2 + corr1 * (1 - mask2)
            corr0 = corr1 * mask1 + corr0 * (1 - mask1)
            return flow0, corr0
        else:
            return flow0


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


class EDBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class EUBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EUBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, feat1, feat0):

        feat = self.up(feat1) + feat0
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
