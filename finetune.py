from __future__ import print_function, absolute_import
import os
import sys
import shutil
import time
import datetime
import copy
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from utils import *
import dataloader

parser = argparse.ArgumentParser(description='Finetuning pre-trained model for image classification')
parser.add_argument('--data', type=str, default='./data/', help="data path")
parser.add_argument('--path', type=str, default='', help="root path to data directory (default: none)")
parser.add_argument('-j', '--workers', type=int, default=20,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--size', type=int, default=224,
                    help="size of an image (default: 224)")
parser.add_argument('--baselr', type=float, default=0.0001,
                    help="initial learning rate for base network  (default: 0.0001)")
parser.add_argument('--fclr', type=float, default=0.001,
                    help="initial learning rate for classification layer (default: 0.001)")
parser.add_argument('-b', '--batch-size', type=int, default=8,
                    help='mini-batch size (default: 8)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay (default: 1e-4)")
parser.add_argument('--gamma', type=float, default=0.1,
                    help="learning rate decay (default: 0.1)")
parser.add_argument('--stepsize', type=int, default=0,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--resume', type=str, default='',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', type=int, default=10, help="print frequency (default: 10)")
parser.add_argument('-a', '--arch', type=str, default='resnet18')
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--save-dir', type=str, default='./checkpoints/')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--start-epoch', type=int, default=0,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--epochs', type=int, default=30,
                    help="number of total epochs to run (default: 30)")
parser.add_argument('--evaluate', type=str, default='',
                    help='evaluate model on validation set (default: none)')
parser.add_argument('--model-zoo', type=str, default='',
                            help='root path to model zoo (default: none)')
def main():
    global args
    args = parser.parse_args()

    if args.model_zoo:
        os.environ['TORCH_MODEL_ZOO'] = args.model_zoo
    
    torch.manual_seed(args.seed)
    best_prec1 = 0
    best_epoch = 0
    #set devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False   
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
    
    #load data
    print("=> Initializing dataset {}".format(args.data))
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(args.size+32),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    if os.path.isfile(os.path.join(args.data, 'train.txt')) and os.path.isfile(os.path.join(args.data, 'val.txt')):
        image_datasets = {x: dataloader.ImageList(os.path.join(args.data, x+'.txt'),
                                                  data_transforms[x], root=args.path)
                          for x in ['train', 'val']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x),
                                                  data_transforms[x], root=args.path)
                          for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers)
                  for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    class_num = len(image_datasets['train'].classes)
    
    #create model
    print("=> Creating model '{}'".format(args.arch))
    
    if args.arch=='mobilenetv2':
        model = MobileNetV2()
        state_dict = torch.load('./models/mobilenetv2.pth.tar');
        model.load_state_dict(state_dict)
    else:
        model = models.__dict__[args.arch](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_num)
    print("Model '{}' with the size of {:.4f}M".format(args.arch ,sum(p.numel() for p in model.parameters())/1000000.0))
    
    #training set
    criterion = nn.CrossEntropyLoss().cuda()
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                        model.parameters())
    optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': model.fc.parameters(), 'lr': args.fclr}
                ], lr=args.baselr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
    if args.stepsize == 0: args.stepsize = args.epochs//3 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("Loading pre-trained model from '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            prec1 = validate(model, criterion, dataloaders['val'], use_gpu)
        else:
            print("No model found at '{}'".format(args.evaluate))
        return
    
    start_time = time.time()
    train_time = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        start_train_time = time.time()
        if args.stepsize > 0: scheduler.step(epoch)
        lr_list = scheduler.get_lr()
        print('>> Epoch#{:d} \t'
              'Base_LR {baselr:.2e} \t'
              'FClayer_LR {fclr:.2e} \t'
              'Decay {wd:.2e} \t'
              'Batch-size {bs:d} \t'.format(
               epoch+1, baselr=lr_list[0], fclr=lr_list[1], wd=args.weight_decay, bs=args.batch_size))
        
        train(epoch, model, criterion, optimizer, dataloaders['train'], use_gpu)
        train_time += round(time.time() - start_train_time)
        prec1 = validate(model, criterion, dataloaders['val'], use_gpu)
        is_best = prec1 > best_prec1
        if is_best: best_epoch = epoch+1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'), osp.join(args.save_dir, 'model_best.pth.tar'))
    print("==> Best Top-1 {:.2f}%, achieved at epoch {}".format(best_prec1, best_epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("==> Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    corrects = AverageMeter()
    
    model.train()
    end = time.time()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        losses.update(loss.item(), labels.size(0))
        _, preds = torch.max(outputs.data, 1)
        correct = float(torch.sum(preds == labels.data))/labels.size(0)
        corrects.update(correct, labels.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {correct.val:.4f} ({correct.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, correct=corrects))

def validate(model, criterion, valloader, use_gpu):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(valloader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        if outputs.size(1) >5:
            top_num = 5
        else:
            top_num = outputs.size(1)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, top_num))
        loss = criterion(outputs, targets)
        losses.update(loss.item(), targets.size(0))
        top1.update(prec1[0], targets.size(0))
        top5.update(prec5[0], targets.size(0))
    test_time=time.time() - end
    print('** Time {:.3f} \tPrec@1 {top1.avg:.3f}  \tPrec@{top_num:d} {top5.avg:.3f}'
              .format(test_time, top_num=top_num, top1=top1, top5=top5))
    return top1.avg
  
if __name__ == '__main__':
    main() 
