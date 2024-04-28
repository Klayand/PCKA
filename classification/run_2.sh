#!/bin/sh

#CUDA_VISIBLE_DEVICES=1 python C100/train_student_dkd.py --checkpoint-dir ./store/ \
#  --gpu-id 1  --i 4 --data /home/sst/dataset/c100 --arch resnet20 --tarch resnet56 --last-encoder transformer
#CUDA_VISIBLE_DEVICES=1 python C100/train_student_dkd.py --checkpoint-dir ./store/ \
#  --gpu-id 1  --i 5 --data /home/sst/dataset/c100 --arch resnet20 --tarch resnet56 --last-encoder mlp
#CUDA_VISIBLE_DEVICES=1 python C100/train_student_dkd.py --checkpoint-dir ./store/ \
#  --gpu-id 1  --i 6 --data /home/sst/dataset/c100 --arch resnet20 --tarch resnet56 --last-encoder conv


CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ \
  --gpu-id 1  --i 3 --data /home/sst/dataset/c100 --arch resnet20_fakd --tarch resnet56 --dirac_ratio 1. --flow-loss-type dist --encoder-type transformer

CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ \
  --gpu-id 1  --i 4 --data /home/sst/dataset/c100 --arch resnet20_fakd --tarch resnet56 --dirac_ratio 1. --flow-loss-type dist --encoder-type mlp

CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ \
  --gpu-id 1  --i 5 --data /home/sst/dataset/c100 --arch resnet20_fakd --tarch resnet56 --dirac_ratio 1. --flow-loss-type dist --encoder-type conv
