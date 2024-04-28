#!/bin/sh
#
#CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ --flow-weight 1. \
#  --gpu-id 1  --i 1 --data /home/sst/dataset/c100 --arch resnet20_fakd --tarch resnet110 --dirac_ratio 1. --flow-loss-type dist \
#  --encoder-type transformer --tcheckpoint /home/sst/product/mdistiller/download_ckpts/cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth
##
#CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ --flow-weight 1. \
#  --gpu-id 1  --i 1 --data /home/sst/dataset/c100 --arch wrn_40_1_fakd --tarch wrn_40_2 --dirac_ratio 1. --flow-loss-type dist \
#  --encoder-type transformer --tcheckpoint /home/sst/product/mdistiller/download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth
#
CUDA_VISIBLE_DEVICES=1 python C100/train_student_fakd_dist_pkd.py --checkpoint-dir ./store/ --flow-weight 1. \
  --gpu-id 1  --i 12 --data /home/sst/dataset/c100 --arch wrn_40_1_fakd --tarch wrn_40_2 --dirac_ratio 1. --flow-loss-type dist \
  --encoder-type transformer --tcheckpoint /home/sst/product/mdistiller/download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth