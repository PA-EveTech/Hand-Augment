from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import argparse
from preprocess_data import process_data
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import models

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='AEMG data training using an RNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_num', metavar='NUM', type=int, default=2,
                    help='which dataset to use (1 or 2)')
parser.add_argument('--gender', metavar='GEN', type=str, default='male',
                    help='which gender in the dataset to use (male or female')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='comma-seperated string of gpu ids to use for acceleration (-1 for cpu only)')
# Model Constructor Arguments
parser.add_argument('--encoder_size', type=int, default=400, help='size of encoder hidden layer')
parser.add_argument('--decode_size', type=int, default=400, help='size of decoder hidden layer')
parser.add_argument('--t_steps', type=int, default=3, help='number of time steps to iterate through RNN')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--cosine', action='store_true', help='use cosine annealing schedule to decay learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

# Model checkpoint flags
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

global best_acc1

args = parser.parse_args()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.dataset != 'xrays':
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        #import pdb
        # measure data loading time
        data_time.update(time.time() - end)

        if '-1' not in args.gpu_ids:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if args.dataset == 'xrays':
            acc1, _ = accuracy(output, target, topk=(1, 1))
        else:
            #pdb.set_trace()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        if args.dataset != 'xrays':
            top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    summary.add_scalar('train acc1', top1.avg, epoch)
    #summary.add_scalar('train acc5', top5.avg, epoch)
    summary.add_scalar('train loss', losses.avg, epoch)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.dataset != 'xrays':
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix="Test: ")
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if '-1' not in args.gpu_ids:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if args.dataset == 'xrays':
                acc1, _ = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            if args.dataset != 'xrays':
                top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        summary.add_scalar('test acc1', top1.avg, epoch)
        #summary.add_scalar('test acc5', top5.avg, epoch)
        summary.add_scalar('test loss', losses.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    args = parser.parse_args()
    best_acc1=0
     
    if torch.cuda.is_available():
        gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_id_list = [-1]

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    summary = SummaryWriter(args.save_path)
    print('==> Output path: {}...'.format(args.save_path))

    assert args.dataset_num in [1, 2], "Dataset number must be either 1 or 2"
    assert args.gender in ['male', 'female'], "Dataset gender must be either male or female"

    if args.gender == 'male':
        dataset, labels = process_data(args.dataset_num, 1)
    else:
        dataset, labels = process_data(args.dataset_num, 0)
    train_set, test_set, train_labels, test_labels = train_test_split(dataset, labels, test_size = 0.2)
    new_train_set = []
    new_test_set = []

    for i in range(train_set.shape[0]):
        new_train_set.append(train_set[i].flatten())
    for i in range(test_set.shape[0]):
        new_test_set.append(test_set[i].flatten())

    new_train_set = torch.utils.data.TensorDataset(torch.Tensor(new_train_set), torch.Tensor(train_labels).long())
    new_test_set = torch.utils.data.TensorDataset(torch.Tensor(new_test_set), torch.Tensor(test_labels).long())

    train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(new_test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = models.__dict__[args.arch](train_data=train_loader, encoder_hidden_size, decoder_hidden_size, T, learning_rate, batch_size)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) 
            checkpoint['state_dict'] = {n.replace('module.', '') : v for n, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'" .format(args.resume))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))
    else:
        print("=> Not using any checkpoint for {} model".format(args.arch))

    #print(model)

    if -1 not in gpu_id_list:
        model = torch.nn.DataParallel(model, device_ids = gpu_id_list)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    if -1 not in gpu_id_list and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    if args.evaluate:
        validate(test_loader, model, criterion, 1, args)

    if args.train:
        for epoch in range(args.epochs):

            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1 = validate(test_loader, model, criterion, epoch, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        
            save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args.save_path)

    summary.close()

model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model.fit(new_train_set, train_labels)
print(model.score(new_test_set, test_labels))