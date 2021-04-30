#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import logging
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import moco.models as models
import moco.swav_models as swav_models
import moco.loader
import moco.builder_distill_swav_small_patch
from moco.dataset import Small_Patch_TSVDataset
from moco.logger import setup_logger

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

swav_model_names = sorted(name for name in swav_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(swav_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output', metavar='DIR',
                    help='path to output folder')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv3_large',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-k', '--key_arch', metavar='ARCH', default='resnet50',
                    choices=swav_model_names,
                    help='key encoder architecture: ' +
                        ' | '.join(swav_model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--distill', default='', type=str, metavar='PATH',
                    help='path to distillation model.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--info', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--distill-t', default=1e-4, type=float,
                    help='softmax temperature for distillation (default: 1e-4)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="moco")
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    main_worker(args, logger)


def main_worker(args, logger):

    # check the PyTorch version
    if not torch.__version__ == '1.6.0':
        logger.info('Wrong PyTorch Version, v-1.6.0 Expected.')

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    logger.info(swav_models.__dict__[args.key_arch].__name__)

    # hidden_dim: resnet50-2048, resnet50w4-8192, resnet50w5-10240
    if args.key_arch == 'resnet50': swav_mlp = 2048
    elif args.key_arch == 'resnet50w2': swav_mlp = 8192
    elif args.key_arch == 'resnet50w4': swav_mlp = 8192
    elif args.key_arch == 'resnet50w5': swav_mlp = 10240

    model = moco.builder_distill_swav_small_patch.MoCo(models.__dict__[args.arch],
                                           swav_models.__dict__[args.key_arch],
                                           args.moco_dim, args.moco_k,
                                           args.moco_m, args.moco_t, mlp=args.mlp,
                                           temp=args.distill_t, swav_mlp=swav_mlp)

    logger.info(model)

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                      device_ids=[args.local_rank],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)

    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    args.lr_mult = args.batch_size / 256
    args.warmup_epochs = 5
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr_mult * args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    # activate the distillation
    if not args.resume and args.distill:
        if os.path.isfile(args.distill):
            logger.info("=> loading distillation checkpoint '{}'".format(args.distill))

            checkpoint = torch.load(args.distill)
            model_checkpoint = model.state_dict()

            # update only the key-encoder
            for key in checkpoint:
                # change param name from 'module.conv1*' ==> 'module.encoder_k.conv1*'
                # exclude the prototype branch
                if not key.startswith('module.prototypes'):
                    model_key = key.replace('module', 'module.encoder_k')
                    model_checkpoint[model_key] = checkpoint[key]
                    logger.info('{} loaded.'.format(model_key))

            model.load_state_dict(model_checkpoint)
            del model_checkpoint
            del checkpoint
            logger.info("=> distillation checkpoint loaded '{}'".format(args.distill))
        else:
            logger.info("wrong distillation checkpoint.")

    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    torch.cuda.empty_cache()

    # Data loading code
    traintsv = os.path.join(args.data, 'train.tsv')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.14, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    small_patch_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = Small_Patch_TSVDataset(traintsv, augmentation, small_patch_augmentation, num_patches=6)
    logger.info('TSV Dataset done.')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(),\
        'Batch size is not divisible by num of gpus.'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, acc1 = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('train_acc1', acc1, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if dist.get_rank() == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.output, '{}_{}_{}_distill_{}-swav-checkpoint_{:04d}.pth.tar'
                                                    .format(args.info, args.epochs, args.arch, args.key_arch, epoch)))
            logger.info('==============> checkpoint saved to {}'
                  .format(os.path.join(args.output, '{}_{}_{}_distill_{}-swav-checkpoint_{:04d}.pth.tar'
                                       .format(args.info, args.epochs, args.arch, args.key_arch, epoch))))


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Small-Acc@1', ':5.2f')
    lr = ValueMeter('LR', ':5.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    model.module.encoder_k.eval()

    end = time.time()

    # check the sanity of key-encoder
    for name, param in model.module.encoder_k.named_parameters():
       if param.requires_grad:
            logger.info("====================> "
                        "Key-encoder Sanity Failed, parameters are not frozen!")

    for i, (images, small_images) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        logit, label, logit_small, label_small, target, small_target\
            = model(img=images, small_img=small_images)

        loss = criterion(logit, label) + criterion(logit_small, label_small)

        # acc1 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, _ = accuracy(logit, target, topk=(1, 5))
        small_acc1, _ = accuracy(logit_small, small_target, topk=(1, 5))

        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(small_acc1[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ValueMeter(object):
    """stores the current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * args.lr_mult
    if epoch < args.warmup_epochs:
        # warm up
        lr = args.lr + (args.lr * args.lr_mult - args.lr) / args.warmup_epochs * epoch
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def soft_cross_entropy(student_logit, teacher_logit):
    '''
    :param student_logit: logit of the student arch (without softmax norm)
    :param teacher_logit: logit of the teacher arch (already softmax norm)
    :return: CE loss value.
    '''
    return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum() / student_logit.shape[0]

if __name__ == '__main__':
    main()
