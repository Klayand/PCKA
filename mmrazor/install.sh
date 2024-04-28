#!/bin/bash

conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
mim install mmcls==1.0.0rc0 mmcv==2.0.0rc3 mmdet==3.0.0rc1 mmengine==0.1.0 mmsegmentation==1.0.0rc1 -i \
   https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops numpy==1.20.0
pip install -v -e .
