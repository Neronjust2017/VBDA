"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../../..')

from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

from modules.kernels import GaussianKernel
from modules.vbda import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier

sys.path.append('.')
import utils
import metrics
import config_bayesian as cfg
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="vbda_ddp", entity="yuhuawei", name="office31_mc_2_2gpu_tran5")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    # wandb config
    # wandb.config = vars(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # load config parameters
    priors = cfg.priors
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model_vbda(args.arch, priors=priors)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch)

    classifier = torch.nn.DataParallel(classifier).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.module.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not args.non_linear
    )

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, mkmmd_loss, optimizer,
              lr_scheduler, epoch, train_ens, args)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          mkmmd_loss: MultipleKernelMaximumMeanDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, num_ens: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    kl_losses = AverageMeter('KL Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mkmmd_loss.train()

    end = time.time()


    for i in range(args.iters_per_epoch):

        t1 = time.time()

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output

        xs_s = torch.cat([x_s for _ in range(num_ens)], dim=0)
        xs_t = torch.cat([x_t for _ in range(num_ens)], dim=0)

        ys_s, fs_s, _kl = model(xs_s)
        ys_t, fs_t, _ = model(xs_t)

        ys_s = ys_s.chunk(num_ens, dim=0)
        fs_s = fs_s.chunk(num_ens, dim=0)
        ys_t = ys_t.chunk(num_ens, dim=0)
        fs_t = fs_t.chunk(num_ens, dim=0)

        ys_s = torch.stack(ys_s, dim=-1)
        fs_s = torch.stack(fs_s, dim=-1)
        ys_t = torch.stack(ys_t, dim=-1)
        fs_t = torch.stack(fs_t, dim=-1)

        ys_s = utils.logmeanexp(F.log_softmax(ys_s, dim=1), dim=2)
        ys_t = utils.logmeanexp(F.log_softmax(ys_t, dim=1), dim=2)

        fs_s = torch.mean(fs_s, dim=2)
        fs_t = torch.mean(fs_t, dim=2)

        cls_loss = F.nll_loss(ys_s, labels_s, reduction='mean')
        transfer_loss = mkmmd_loss(fs_s, fs_t)
        kl_loss = _kl

        # beta
        loss = cls_loss + transfer_loss * args.alph * 0

        cls_acc = accuracy(torch.exp(ys_s), labels_s)[0]
        tgt_acc = accuracy(torch.exp(ys_t), labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        # kl_losses.update(kl_loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        t2 = time.time()
        step_time = t2 - t1

        wandb.log(
            {
                "step_time": step_time
            }
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
    parser.add_argument('--alph', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--beta', default=1., type=float,
                        help='the trade-off hyper-parameter for kl loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

