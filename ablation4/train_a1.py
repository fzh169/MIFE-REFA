import os
import datetime
import argparse

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.distributed.launch

import utility
from losses.loss import Loss
from datasets import Vimeo90K_interp, ATD12K_interp
from test import Middlebury_other, test
from models.mymodel import MyModel

# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train.py
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py

def parse_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Hardware Setting
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank')
    parser.add_argument('--world_size', default=-1, type=int, help='world size')

    # Directory Setting
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/train_10k_540p/')
    parser.add_argument('--out_dir', type=str, default='./out')
    parser.add_argument('--test_input', type=str, default='/root/autodl-tmp/middlebury_others/input')
    parser.add_argument('--test_gt', type=str, default='/root/autodl-tmp/middlebury_others/gt')
    parser.add_argument('--weights_dir', type=str, default='./real_time.pth')
    # parser.add_argument('--weights_dir', type=str, default='')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=50, help='max epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--loss', type=str, default='1*rec+0.1*ms', help='loss function configuration')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, help='warmup learning rate')
    parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'RMSprop', 'Adam', 'ADAMax'), help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    save_path = './weights/' + timestamp

    parser.add_argument('--save_path', default=save_path, help='the output dir of weights')
    parser.add_argument('--log', default=save_path + '/log.txt', help='the log file in training')
    parser.add_argument('--arg', default=save_path + '/args.txt', help='the args used')

    args = parser.parse_args()

    return args


class Trainer:
    def __init__(self, args, train_loader, test_loader, sampler, my_model, my_loss, start_epoch=1):
        self.args = args
        self.local_rank = args.local_rank
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.sampler = sampler
        self.model = my_model
        self.loss = my_loss
        self.current_epoch = start_epoch
        self.save_path = args.save_path

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer, self.max_step)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        self.result_dir = args.save_path + '/results'
        self.ckpt_dir = args.save_path + '/checkpoints'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=True)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logfile = open(args.log, 'w')

        if args.local_rank == 0:
            self.model.eval()
            self.best_psnr = self.test_loader.test(self.model, self.result_dir,
                                                   output_name=str(self.current_epoch).zfill(3),
                                                   file_stream=self.logfile)
            # self.best_psnr = 0

    def train(self):
        self.model.train()
        self.sampler.set_epoch(self.current_epoch)
        # self.scheduler.step_update((self.current_epoch - 1) * self.max_step - 1)
        for batch_idx, (frame0, frame1, frame2) in enumerate(self.train_loader):

            frame0 = frame0.cuda()
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()

            self.optimizer.zero_grad()
            output = self.model(frame0, frame2)
            loss = self.loss(output, [frame0, frame1, frame2])
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0 and self.local_rank == 0:
                torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict()},
                           self.ckpt_dir + "/real_time.pth")
                utility.print_and_save('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ', '[' + str(
                    self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(
                    self.max_step) + ']', 'train loss: ', loss.item()), self.logfile)

            self.scheduler.step_update((self.current_epoch - 1) * self.max_step + batch_idx)

        self.current_epoch += 1

        if self.local_rank == 0:
            utility.print_and_save('===== current lr: %f =====' % (self.optimizer.param_groups[0]['lr']), self.logfile)

        dist.barrier()

    def test(self):
        utility.print_and_save('Testing...', self.logfile)
        self.model.eval()
        tmp_psnr = self.test_loader.test(self.model, self.result_dir, output_name=str(self.current_epoch).zfill(3),
                                         file_stream=self.logfile)
        if tmp_psnr > self.best_psnr:
            self.best_psnr = tmp_psnr
        
        if self.current_epoch % 10 == 0:
            torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict()},
                       self.ckpt_dir + '/model_epoch_' + str(self.current_epoch).zfill(3) + '.pth')

    def terminate(self):
        return self.current_epoch > self.args.epochs

    def close(self):
        self.logfile.close()


def main():
    args = parse_args()

    if'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank == 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)

        with open(args.log, 'w') as f:
            f.close()
        with open(args.arg, 'w') as f:
            print(args)
            print(args, file=f)
            f.close()

    torch.cuda.set_device(args.local_rank)

    # dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='10000')
    # dist.init_process_group(backend="nccl", init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
    dist.init_process_group(backend="nccl", world_size=args.world_size)
    dist.barrier()

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    train_dataset = ATD12K_interp(args.data_dir, random_crop=(512, 512), resize=None,
                                    augment_s=True, augment_t=True)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=24, pin_memory=True,
                              sampler=sampler)

    test_dataset = Middlebury_other(args.test_input, args.test_gt)

    device = torch.device('cuda', args.local_rank)
    model = MyModel(args).to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print('===============================')
    print("# of model parameters is: " + str(utility.count_network_parameters(model)))

    if args.weights_dir != "":
        if os.path.exists(args.weights_dir):
            weights_dict = torch.load(args.weights_dir, map_location=device)
            weights_dict = weights_dict['state_dict']
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    else:
        checkpoint_dir = args.save_path + '/checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_dir + '/real_time.pth'
        if args.local_rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # test(model)

    loss = Loss(args)

    my_trainer = Trainer(args, train_loader, test_dataset, sampler, model, loss)

    while not my_trainer.terminate():
        my_trainer.train()
        if args.local_rank == 0:
            my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
