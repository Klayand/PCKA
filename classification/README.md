## What Role Does Data Augmentation Play in Knowledge Distillation ?

- This project provides source code for our Joint Data Augmentation and Cosine Confidence Distillation (JDA and CCD).
- This paper is publicly available at the official ACCV proceedings.
- - 🏆 __SOTA of Knowledge Distillation for student ResNet-18 trained by teacher ResNet-34 on ImageNet.

### Installation

#### Requirements

- pytorch
- numpy
- Ubuntu 20.04 LTS

1. please install create environment first.

```bash
conda create -n torch python==3.7.10
```

2. then activate this environment.

```bash
source activate torch
```

3. then install pytorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


### Perform Offline KD experiments on CIFAR-100 dataset
#### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

- download to your folder, named [cifar100_folder]

#### Teacher checkpoint

CKPT: [download](https://github.com/shaoshitong/torchdistill/releases/tag/v0.3.2)

- download to your floder, name [ckpt_folder]

#### Train KD+JDA and CCD+JDA

1. Modify the paths of the teacher pre-training files and datasets in these two files : multirunc100_jda.sh, multirunc100_jda_ccd.sh

2. run them

```bash
bash multirunc100_jda.sh
bash multirunc100_jda_ccd.sh
```
### Perform Offline KD experiments on ImageNet dataset
#### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

- Create a datasets subfolder and a symlink to the ImageNet dataset

Folder of ImageNet Dataset:
```
data/ImageNet
├── train
├── val
```

#### Teacher checkpoint

The pre-trained backbone weights of ResNet-34 follow the [resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth) downloaded from the official PyTorch. 

#### Train KD+JDA and CCD+JDA
1. Modify the paths of the teacher pre-training files and datasets in this file : train_student_imagenet.py

2. run this py format file

```bash
python train_student_imagenet.py
python train_student_imagenet.py --ccd
```

### Code References

Our code is based on [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) (e.g., SPKD, KD, SSKD, CRD) and [HSAKD](https://github.com/winycg/HSAKD) (e.g., the training framework).
