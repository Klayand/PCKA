import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import SubsetRandomSampler
import os
import shutil
import argparse
import numpy as np
import models
import torchvision
import torchvision.transforms as transforms
from C10.utils import cal_param_size, cal_multi_adds
import losses
from utils import *
from data.datasets import PolicyDatasetImageNet
from data.cached_image_folder import CachedImageFolder
from losses import PKDLoss

from bisect import bisect_right
import time
import math
from utils.weight_dacay import add_weight_decay

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/imagenet/SJTUCCC/zhangyijia_1/AI_Dev_meetup_2022/datasets/imagenet', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet18_imagenet_fakd', type=str, help='student network architecture')
parser.add_argument('--tarch', default='resnet50_imagenet_timm', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='./resnet50-19c8e357.pth', type=str,
                    help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--schedule', default=[2.4,0.97], type=list, help='lr-scheduler')
parser.add_argument('--layer-weight', default=[0,0,0,1], type=list, help='the layer weight of network')
parser.add_argument('--flow-loss-type', type=str, default=None, help='loss type of flow')
parser.add_argument('--encoder-type', type=str, default=None, help='encoder type of flow')
parser.add_argument('--flow_weight', type=float, default=1, help='weight for Flow Align distillation')
parser.add_argument('--dirac_ratio', type=float, default=1.0)
parser.add_argument('--warmup-epoch', default=3, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=450, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
parser.add_argument('--accumulate-step', type=int, default=1, help='accumulate the grad')
parser.add_argument('--num-workers', type=int, default=4, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=1, help='temperature for KD distillation')
parser.add_argument('--kd_alpha', type=float, default=0.5, help='alpha for KD distillation')
parser.add_argument('--cutmix', action='store_true', help="if use CutMix")
parser.add_argument('--zip', action='store_true', help="if use zip to read image")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def smooth_label(gold, n_class=1000, smoothing=0.1):
    one_hot = torch.full(size=(gold.shape[0],n_class), fill_value=smoothing / (n_class - 1)).to(gold.device)  # 0.0111111
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1).long(), value=1. - smoothing)  # 0.9
    return one_hot

import copy
class EMAMODEL(object):
    def __init__(self,model):
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self,model=None,decay_rate=0.9999):
        for param, ema_param in zip(model.parameters(),self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)
    
    @torch.no_grad()
    def ema_swap(self,model=None):
        for param,ema_param in zip(self.ema_model.parameters(),model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp
            
    def __call__(self,*args,**kwargs):
        return self.ema_model(*args,**kwargs)
    
    def eval(self):
        self.ema_model.eval()
        
    def train(self):
        self.ema_model.train()

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
                   'arch' + '_' + args.arch + '_' + \
                   'dataset' + '_' + args.dataset + '_' + \
                   'seed' + str(args.manual_seed) + '.txt'

    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_' + \
                   'arch' + '_' + args.arch + '_' + \
                   'dataset' + '_' + args.dataset + '_' + \
                   'seed' + str(args.manual_seed)

    args.traindir = os.path.join(args.data, 'train')
    args.valdir = os.path.join(args.data, 'val')

    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')
    print('==> Building model..')
    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))
    checkpoint = torch.load(args.tcheckpoint, map_location='cpu')
    torch.cuda.set_device(args.gpu)

    manual_seed = gpu + args.manual_seed
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    
    num_classes = 1000
    model = getattr(models, args.arch)
    teacher_features = [128, 256, 512]
    teacher_sizes = [28, 14, 7]
    net = model(num_classes=num_classes,
                teacher_features=teacher_features,
                teacher_sizes=teacher_sizes,
                dirac_ratio=args.dirac_ratio,
                weight=args.layer_weight,
                flow_loss_type=args.flow_loss_type,
                encoder_type = args.encoder_type
                ).cuda(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],find_unused_parameters=True)

    tmodel = getattr(models, args.tarch)
    tnet = tmodel(num_classes=num_classes).cuda(args.gpu)
    tnet = torch.nn.parallel.DistributedDataParallel(tnet, device_ids=[args.gpu])
    if "timm" not in args.tarch:
        tnet.module.load_state_dict(checkpoint)
    tnet.eval()
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.batch_size = int(args.batch_size // (args.accumulate_step) + args.batch_size % (args.accumulate_step))
    print(f"local batch size is {args.batch_size}")
    args.workers = int(args.num_workers)
    cudnn.benchmark = True
    cudnn.fastest = True
    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = losses.DISTLoss()
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    ema_net = EMAMODEL(net)
    optimizer = optim.AdamW(add_weight_decay(trainable_list,weight_decay=args.weight_decay,lr=args.init_lr),
                          lr=args.init_lr, weight_decay=args.weight_decay)

    if args.resume:
        print('load intermediate weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        ema_net.ema_model.module.load_state_dict(checkpoint['ema_net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    if args.zip:
        ann_file = "train_map.txt"
        prefix = "train.zip@/"
        train_set = CachedImageFolder(args.traindir, ann_file, prefix, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), cache_mode='part')
        indices = np.arange(dist.get_rank(), len(train_set), dist.get_world_size())
        train_sampler = SubsetRandomSampler(indices)
    else:
        l_trans,r_trans = build_train_transforms(color_jitter=0.05)
        train_set = torchvision.datasets.ImageFolder(args.traindir,l_trans)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,shuffle=True)
        
    test_set = torchvision.datasets.ImageFolder(
        args.valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    trainloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, sampler=train_sampler,drop_last=True,collate_fn=fast_collate)
    trainloader = DataPrefetcher(trainloader,transforms=r_trans,mixup_transform=None)

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    
    def train(epoch, criterion_list, optimizer, iter_number):
        
        trainloader.loader.sampler.set_epoch(epoch)
        tnet.train()
        train_loss = 0.
        train_loss_div = 0.
        top1_num = 0
        top5_num = 0
        total = 0
        if epoch >= args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args)
        start_time = time.time()
        criterion_div = criterion_list[1]
        net.train()
        for batch_idx, (input, target) in enumerate(trainloader):
            batch_start_time = time.time()
            input = input.float().cuda(non_blocking=True)
            if input.ndim == 5:
                b, m, c, h, w = input.shape
                input = input.view(-1, c, h, w)
                target = target.view(-1)
            target = target.cuda(non_blocking=True)
            one_hot_target = smooth_label(target)
            if epoch < args.warmup_epoch:
                lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))
            if iter_number % args.accumulate_step == 0:
                optimizer.zero_grad()
                iter_number = 0

            with torch.no_grad():
                t_logits = tnet(input)
                t_features = [None] * 3
            with torch.cuda.amp.autocast(enabled=True):
                s_features, logits, flow_loss = net(input, t_features + [t_logits], is_feat=True, target=one_hot_target,
                                                    epoch=epoch)
                logits = logits.float()
            
            loss_div = torch.tensor(0.).cuda()
            loss_div = loss_div + flow_loss
            loss = loss_div / args.accumulate_step
            scaler.scale(loss).backward()
            if (iter_number + 1) % args.accumulate_step == 0:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 20.)
                scaler.step(optimizer)
                scaler.update()
                ema_net.ema_step(net)
                # optimizer.step()
            iter_number = iter_number + 1
            train_loss += loss.item() / len(trainloader)
            train_loss_div += loss_div.item() / len(trainloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)
            if batch_idx % 100 == 0 and gpu == 0:
                print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Loss: {:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(trainloader), lr, time.time() - batch_start_time, train_loss,
                    (top1_num / (total)).item()))
        if gpu==0:
            with open(args.log_txt, 'a+') as f:
                f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                        '\n train_loss:{:.5f}\t train_loss_div:{:.5f}'
                        'Train Top-1 class_accuracy: {}\nTrain Top-5 class_accuracy: {}\n'
                        .format(epoch, lr, time.time() - start_time,
                                train_loss, train_loss_div, str((top1_num / (total)).item()),
                                str((top5_num / (total)).item())))

    def test(epoch, criterion_cls, _net, apply_sampling=False):
        _net.eval()
        if apply_sampling:
            sampling_best_acc = 0
            for sampling in [1,2,4,8]:
                test_loss_cls = 0.
                top1_num = 0
                top5_num = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, target) in enumerate(testloader):
                        batch_start_time = time.time()
                        input, target = inputs.cuda(), target.cuda()
                        output = _net(input, inference_sampling=sampling)
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        top1, top5 = correct_num(logits, target, topk=(1, 5))
                        top1_num += top1
                        top5_num += top5
                        total += target.size(0)
                        if gpu==0:
                            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                                epoch, batch_idx, len(testloader), time.time() - batch_start_time,
                                (top1_num / (total)).item()))
                    if gpu == 0:
                        with open(args.log_txt, 'a+') as f:
                            f.write(
                                'test epoch:{}\t sampling:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                                    .format(epoch, sampling, test_loss_cls, str((top1_num / (total)).item()),
                                            str((top5_num / (total)).item())))
                        print('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                            .format(epoch, test_loss_cls, str((top1_num / (total)).item()),
                                    str((top5_num / (total)).item())))
                    sampling_best_acc = max((top1_num / (total)).item(), sampling_best_acc)
            return sampling_best_acc
        else:
            test_loss_cls = 0.
            top1_num = 0
            top5_num = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, target) in enumerate(testloader):
                    batch_start_time = time.time()
                    input, target = inputs.cuda(), target.cuda()
                    output = _net(input)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    top1, top5 = correct_num(logits, target, topk=(1, 5))
                    top1_num += top1
                    top5_num += top5
                    total += target.size(0)

                    print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                        epoch, batch_idx, len(testloader), time.time() - batch_start_time, (top1_num / (total)).item()))
                with open(args.log_txt, 'a+') as f:
                    f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                            .format(epoch, test_loss_cls, str((top1_num / (total)).item()),
                                    str((top5_num / (total)).item())))
                print('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                      .format(epoch, test_loss_cls, str((top1_num / (total)).item()), str((top5_num / (total)).item())))

            return float((top1_num / (total)).item())

    if args.evaluate:
        print('load pre-trained weights from: {}'.format(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        # ema_net.ema_swap(net)
        test(start_epoch, criterion_cls, net, apply_sampling=True)
        # ema_net.ema_swap(net)
    else:
        iter_number = 0
        best_acc = 0.
        # acc = test(0,criterion_cls,tnet)
        # print("Teacher Acc is: ",acc)
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer, iter_number)         
            # ema_net.ema_swap(net)
            # acc = test(epoch, criterion_cls, net, apply_sampling=True)
            # ema_net.ema_swap(net)
            _ = test(epoch, criterion_cls, net, apply_sampling=True)  
            if args.rank == 0:
                state = {
                    'net': net.module.state_dict(),
                    'ema_net': ema_net.ema_model.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'scaler': scaler.state_dict(),
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
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        ema_net.ema_model.module.load_state_dict(checkpoint['ema_net'])
        start_epoch = checkpoint['epoch']

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + args.log_txt + ' ' + args.checkpoint_dir)


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


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch * all_iters_per_epoch) / (
                args.warmup_epoch * all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * (args.schedule[1] ** (epoch / args.schedule[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


if __name__ == '__main__':
    main()
