import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os, sys

print(os.getcwd())
sys.path.append(os.path.join(os.getcwd()))
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from data.datasets import PolicyDatasetC10, PolicyDatasetC100, RADatasetC100, AADatasetC100
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count
from bisect import bisect_right
import time
import math, losses

scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='/home/sst/dataset/c100/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_2_fakd', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_2', type=str, help='teacher network architecture')
parser.add_argument('--checkpoint', default='/home/sst/product/CCD/store/train_student_fakd_tarch_wrn_40_2_arch_wrn_16_2_fakd_dataset_cifar100_seed0/wrn_16_2_fakd_best.pth.tar', type=str,
                    help='student trained weight')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--sampling', type=int, default=1, help='the number of sampling steps')
parser.add_argument('--num-workers', type=int, default=4, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--i', type=int, default=1, help='the index of training samples ')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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
# trainset = RADatasetC100(trainset)
testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))
if args.tarch == "wrn_40_2":
    teacher_features = [16 * 2, 32 * 2, 64 * 2]
    teacher_sizes = [32, 16, 8]
else:
    raise NotImplementedError

checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

model = getattr(models, args.arch)
net = model(num_classes=num_classes, teacher_features=teacher_features, teacher_sizes=teacher_sizes).cuda()
net.load_state_dict(checkpoint['net'], strict=False)
net.eval()
net = torch.nn.DataParallel(net)

ss_logits, _ = net(torch.randn(2, 3, 32, 32),
                      [torch.randn(2, teacher_features[0], teacher_sizes[0], teacher_sizes[0]),
                       torch.randn(2, teacher_features[1], teacher_sizes[1], teacher_sizes[1]),
                       torch.randn(2, teacher_features[2], teacher_sizes[2], teacher_sizes[2])])
num_auxiliary_branches = len(ss_logits)
cudnn.benchmark = True


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
            output = net(input,inference_sampling=args.sampling)
            if isinstance(output,tuple):
                logits = output[0]
            else:
                logits = output
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = loss_cls + criterion_cls(logits, target)

            test_loss_cls += loss_cls.item() / len(testloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time() - batch_start_time, (top1_num / (total)).item()))

    return round((top1_num / (total)).item(), 4)


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    print('Evaluate Student:')
    acc = test(0, criterion_cls, net)
    print('Student Acc:', acc)