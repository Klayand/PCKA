import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os, sys

if os.getcwd().split("/")[-1] == "C100":
    os.chdir("../")
    sys.path.append(os.getcwd())
else:
    sys.path.append(os.getcwd())
print(os.getcwd())

import shutil
import argparse
import numpy as np
import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds
from data.datasets import PolicyDatasetC10, PolicyDatasetC100, ContrastiveDataset
from fvcore.nn import flop_count
from bisect import bisect_right
import time
import math, losses

scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_2_crd', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_2_crd', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='./checkpoint/wrn_40_2.pth', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 180, 210], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t', help='SGDR T_0')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd-T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--kd-alpha', type=float, default=0.5, help='temperature for KD distillation')
parser.add_argument('--jda', type=bool, default=True, help='if use data augmentation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='network architecture')
parser.add_argument('--i', type=int, default=1, help='few-shot ratio of training samples ')
parser.add_argument('--flops-calculate', action='store_true', help='count the number of floating point operations')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = 'result/' + 'tarch' + '_' + args.tarch + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + f'_spkd_{args.i}' + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'tarch' + '_' + args.tarch + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)

num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))
trainset = PolicyDatasetC100(trainset)
trainset = ContrastiveDataset(trainset, 16384, 'exact', 1.0, num_classes=num_classes)

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                         pin_memory=(torch.cuda.is_available()))
print('==> Building model..')
print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))
checkpoint = torch.load(args.tcheckpoint, map_location=torch.device('cpu'))
model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)
tmodel = getattr(models, args.tarch)
tnet = tmodel(num_classes=num_classes).cuda()
tnet.load_state_dict(checkpoint['state_dict'], strict=False)
tnet.eval()
tnet = torch.nn.DataParallel(tnet)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch * all_iters_per_epoch) / (
                    args.warmup_epoch * all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.train()
    for batch_idx, (input, target, pos_idx, contrast_idx) in enumerate(trainloader):
        batch_start_time = time.time()
        input = input.float().cuda()
        if input.ndim == 5:
            b, m, c, h, w = input.shape
            input = input.view(-1, c, h, w)
            target = target.view(-1)
            pos_idx = pos_idx.view(-1)
            b, m, n = contrast_idx.shape
            contrast_idx = contrast_idx.view(-1, n)

        target = target.cuda()
        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            crd_s_out, logits = net(input)
            crd_s_out, logits = crd_s_out.float(), logits.float()
        with torch.no_grad():
            crd_t_out, t_logits = tnet(input)
        loss_div = torch.tensor(0.).cuda()
        loss_div = loss_div + criterion_div(crd_s_out, crd_t_out, pos_idx, contrast_idx, logits, t_logits, target)
        loss = loss_div
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)

        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time() - batch_start_time, (top1_num / (total)).item()))
    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_div:{:.5f}'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_div))


def test(epoch, criterion_cls, net):
    global best_acc
    test_loss_cls = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            input, target = inputs.cuda(), target.cuda()
            sf4, logits = net(input)
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = loss_cls + criterion_cls(logits, target)

            test_loss_cls += loss_cls.item() / len(testloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time() - batch_start_time, (top1_num / (total)).item()))
        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\t test_acc:{:.5f}\n'
                    .format(epoch, test_loss_cls, (top1_num / (total)).item()))

    return round((top1_num / (total)).item(), 4)


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = losses.CRDLoss(
        input_size=128,
        output_size=50000,
        num_negative_samples=16384,
        num_samples=50000,
        temperature=0.07,
        momentum=0.5,
        eps=0.0000001, )

    if args.evaluate:
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls, net)
    else:
        print('Evaluate Teacher:')
        acc = test(0, criterion_cls, tnet)
        print('Teacher Acc:', acc)

        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.cuda()

        if args.flops_calculate:
            input, target, pos_idx, contrast_idx = next(iter(trainloader))
            if input.ndim == 5:
                b, m, c, h, w = input.shape
                input: torch.Tensor = input.view(-1, c, h, w).cuda()
                pos_idx = pos_idx.view(-1)
                b, m, n = contrast_idx.shape
                contrast_idx = contrast_idx.view(-1, n)
            target = target.cuda()
            target = target.view(-1)
            tnet = tnet.module
            net = net.module


            class Analysis(nn.Module):
                def __init__(self, net, tnet, dis):
                    super(Analysis, self).__init__()
                    self.net = net
                    self.tnet = tnet
                    self.dis = dis

                def forward(self, data, target, pos_idx, contrast_idx):
                    with torch.no_grad():
                        crd_t_out, tlogit = self.tnet(data)
                    crd_s_out, logit = self.net(data)
                    return self.dis(crd_s_out, crd_t_out, pos_idx, contrast_idx, logit, tlogit, target)


            model = Analysis(net, tnet, criterion_div).cuda()
            flops = flop_count(model, (input, target, pos_idx, contrast_idx))
            unsupported_operator_number = sum(flops[1].values())
            supported_flops = sum(flops[0].values())
            print("flops", supported_flops, "count", unsupported_operator_number)  # 0,32
            begin = time.time()
            for i in range(5000):
                model(input, target, pos_idx, contrast_idx)
            end = time.time()
            use_time = (end - begin) /5000
            print("use time: {}h,{}m,{}s".format(use_time // 3600, use_time % 3600 // 60, use_time % 60))
            """
            flops 111.512190976 count 115
            use time: 0.0h,0.0m,0.038056988716125485s
            """
            exit(-1)
        if args.resume:
            print('load pre-trained weights from: {}'.format(
                os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, net)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)
