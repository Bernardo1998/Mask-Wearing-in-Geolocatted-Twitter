"""
created by: Donghyeon Won
Modified by: Xiaofeng Lin
Modified codes from
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://github.com/pytorch/examples/tree/master/imagenet
    https://github.com/wondonghyeon/protest-detection-violence-estimation
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from util import MaskDataset, modified_resnet50, AverageMeter, Lighting


# for indexing output of the model
mask_idx = Variable(torch.LongTensor([0]))
visattr_idx = Variable(torch.LongTensor(range(1,4)))#Change here when attribute number changes
best_loss = float("inf")


def calculate_loss(output, target, criterions, weights = [1, 5]):
    """Calculate loss"""
    # number of mask images
    N_mask= int(target['mask'].data.sum())
    batch_size = len(target['mask'])

    if N_mask == 0:
        # if no mask related image in target
        outputs = [None]
        # mask output
        outputs[0] = output.index_select(1, mask_idx)
        targets = [None]
        # mask target
        targets[0] = target['mask'].float()
        losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(1)]
        scores = {}
        scores['mask_acc'] = accuracy_score(torch.Tensor.cpu((outputs[0]).data.round()), torch.Tensor.cpu(targets[0].data))
        scores['visattr_acc'] = 0
        return losses, scores, N_mask

    # used for filling 0 for non-mask images
    not_mask_mask = (1 - target['mask']).bool()

    outputs = [None] * 4 #Change here when attribute number changes
    # mask output
    outputs[0] = output.index_select(1, mask_idx)
    # visual attribute output
    outputs[1] = output.index_select(1, visattr_idx)
    outputs[1].masked_fill_(not_mask_mask.repeat(1, 3),0)#Change here when attribute number changes


    targets = [None] * 4#Change here when attribute number changes

    targets[0] = target['mask'].float()
    targets[1] = target['visattr'].float()

    scores = {}
    # mask accuracy for this batch
    scores['mask_acc'] = accuracy_score(torch.Tensor.cpu((outputs[0]).data.round()), torch.Tensor.cpu(targets[0].data))
    # mean accuracy for visual attribute for this batch
    comparison = (outputs[1].data.round() == targets[1].data)
    comparison.masked_fill_(not_mask_mask.repeat(1,3).data,0)#Change here when attribute number changes
    n_right = comparison.float().sum()
    mean_acc = n_right / float(N_mask*3)#Change here when attribute number changes
    scores['visattr_acc'] = mean_acc

    # return weighted loss
    losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(len(criterions))]

    return losses, scores, N_mask


def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_mask = AverageMeter()
    loss_v = AverageMeter()
    mask_acc = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        output = model(input_var)

        losses, scores, N_mask = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        loss = 0
        loss_types = ['mask','visattr']
        for ind, l in enumerate(losses):
            loss += l
        # back prop
        loss.backward()
        optimizer.step()

        if N_mask:
            loss_mask.update(losses[0].item(), input.size(0))
            loss_v.update(loss.item() - losses[0].item(), N_mask)
        else:
            # when there is no mask image in the batch
            mask.update(losses[0].item(), input.size(0))
        loss_history.append(loss.item())
        mask_acc.update(scores['mask_acc'], input.size(0))
        visattr_acc.update(scores['visattr_acc'], N_mask)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'mask {mask_acc.val:.3f} ({mask_acc.avg:.3f})  '
                  'Vis Attr {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss_val=loss_mask.val + loss_v.val,
                   loss_avg = loss_mask.avg + loss_v.avg,
                   mask_acc = mask_acc,
                   visattr_acc = visattr_acc))

    return loss_history

def validate(val_loader, model, criterions, epoch):
    """Validating"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_mask = AverageMeter()
    loss_v = AverageMeter()
    mask_acc = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        input_var = Variable(input)

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        output = model(input_var)

        losses, scores, N_mask = calculate_loss(output, target_var, criterions)
        loss = 0
        loss_types = ['mask','visattr']
        for ind, l in enumerate(losses):
            loss += l

        if N_mask:
            loss_mask.update(losses[0].item(), input.size(0))
            loss_v.update(loss.item() - losses[0].item(), N_mask)
        else:
            # when no mask images
            loss_mask.update(losses[0].item(), input.size(0))
        loss_history.append(loss.item())
        mask_acc.update(scores['mask_acc'], input.size(0))
        visattr_acc.update(scores['visattr_acc'], N_mask)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'mask Acc {mask_acc.val:.3f} ({mask_acc.avg:.3f})  '
                  'Vis Attr Acc {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss_val =loss_mask.val + loss_v.val,
                   loss_avg = loss_mask.avg + loss_v.avg,
                   mask_acc = mask_acc,
                   visattr_acc = visattr_acc))

    print(' * Loss {loss_avg:.3f} mask Acc {mask_acc.avg:.3f} '
          'Vis Attr Acc {visattr_acc.avg:.3f} '
          .format(loss_avg = loss_mask.avg + loss_v.avg,
                  mask_acc = mask_acc,
                  visattr_acc = visattr_acc))
    return loss_mask.avg + loss_v.avg, loss_history

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
    lr = args.lr * (0.4 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoints"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global best_loss
    loss_history_train = []
    loss_history_val = []
    data_dir = args.data_dir
    img_dir_train = os.path.join(data_dir, "img/train_img")
    img_dir_val = os.path.join(data_dir, "img/test_img")
    txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_val = os.path.join(data_dir, "annot_test.txt")

    # load pretrained resnet50 with a modified last fully connected layer, 
    model = modified_resnet50()

    # we need three different criterion for training
    criterion_mask = nn.BCELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_mask, criterion_visattr]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]
    # we are not training the frozen layers
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
                        parameters, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
                        )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            loss_history_train = checkpoint['loss_history_train']
            loss_history_val = checkpoint['loss_history_val']
            if args.change_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    train_dataset = MaskDataset(
                        txt_file = txt_file_train,
                        img_dir = img_dir_train,
                        transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness = 0.4,
                                    contrast = 0.4,
                                    saturation = 0.4,
                                    ),
                                transforms.ToTensor(),
                                Lighting(0.1, eigval, eigvec),
                                normalize,
                        ]))
    val_dataset = MaskDataset(
                    txt_file = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        loss_history_train_this = train(train_loader, model, criterions,
                                        optimizer, epoch)
        loss_val, loss_history_val_this = validate(val_loader, model,
                                                   criterions, epoch)
        loss_history_train.append(loss_history_train_this)
        loss_history_val.append(loss_history_val_this)

        # loss = loss_val.avg

        is_best = loss_val < best_loss
        if is_best:
            print('best model!!')
        best_loss = min(loss_val, best_loss)


        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }, is_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "mask_img",
                        help = "directory path to masks",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    args = parser.parse_args()

    if args.cuda:
        mask_idx = mask_idx.cuda()
        visattr_idx = visattr_idx.cuda()
        
    main()
