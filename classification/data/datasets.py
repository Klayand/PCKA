import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
import random
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


class BaseDatasetWrapper(Dataset):
    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target = self.org_dataset.__getitem__(index)
        return sample, target

    def __len__(self):
        return len(self.org_dataset)


def rotate_with_fill(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)


def shearX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def shearY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def translateX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                         fillcolor=fillcolor)


def translateY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                         fillcolor=fillcolor)


def rotate(img, magnitude, fillcolor):
    return rotate_with_fill(img, magnitude)


def color(img, magnitude, fillcolor):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def posterize(img, magnitude, fillcolor):
    return ImageOps.posterize(img, magnitude)


def solarize(img, magnitude, fillcolor):
    return ImageOps.solarize(img, magnitude)


def contrast(img, magnitude, fillcolor):
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))


def sharpness(img, magnitude, fillcolor):
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def brightness(img, magnitude, fillcolor):
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def autocontrast(img, magnitude, fillcolor):
    return ImageOps.autocontrast(img)


def equalize(img, magnitude, fillcolor):
    return ImageOps.equalize(img)


def invert(img, magnitude, fillcolor):
    return ImageOps.invert(img)


class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        ranges = {
            'shearX': np.linspace(0, 0.3, 10),
            'shearY': np.linspace(0, 0.3, 10),
            'translateX': np.linspace(0, 150 / 331, 10),
            'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10),
            'color': np.linspace(0.0, 0.9, 10),
            'posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0.0, 0.9, 10),
            'sharpness': np.linspace(0.0, 0.9, 10),
            'brightness': np.linspace(0.0, 0.9, 10),
            'autocontrast': [0] * 10,
            'equalize': [0] * 10,
            'invert': [0] * 10
        }

        func = {
            'shearX': shearX,
            'shearY': shearY,
            'translateX': translateX,
            'translateY': translateY,
            'rotate': rotate,
            'color': color,
            'posterize': posterize,
            'solarize': solarize,
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'autocontrast': autocontrast,
            'equalize': equalize,
            'invert': invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        label = 0
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1, self.fillcolor)
            label = 1
        return img, label


class PolicyDatasetC10(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.5):
        super(PolicyDatasetC10, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        print("the probability of CIFAR-10 is {}".format(p))
        org_dataset.transform = None
        self.policies = [
            SubPolicy(p, 'invert', 7),
            SubPolicy(p, 'rotate', 2),
            SubPolicy(p, 'shearY', 8),
            SubPolicy(p, 'posterize', 9),
            SubPolicy(p, 'autocontrast', 8),
            SubPolicy(p, 'color', 3),
            SubPolicy(p, 'sharpness', 9),
            SubPolicy(p, 'equalize', 5),
            SubPolicy(p, 'contrast', 7),
            SubPolicy(p, 'translateY', 3),
            SubPolicy(p, 'brightness', 6),
            SubPolicy(p, 'solarize', 2),
            SubPolicy(p, 'translateX', 3),
            SubPolicy(p, 'shearX', 8),
        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target = super(PolicyDatasetC10, self).__getitem__(index)
        policy_index = torch.zeros(self.policies_len).float()
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
            policy_index[i] = label
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([
            sample,
            new_sample,
        ])
        return sample, target



class PolicyDatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.3):
        super(PolicyDatasetC100, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None
        self.org_dataset=org_dataset
        print("the probability of CIFAR-100 is {}".format(p))
        self.policies = [
            SubPolicy(p, 'autocontrast', 2),
            SubPolicy(p, 'contrast', 3),
            SubPolicy(p, 'posterize', 0),
            SubPolicy(p, 'solarize', 4),

            SubPolicy(p, 'translateY', 8),
            SubPolicy(p, 'shearX', 5),
            SubPolicy(p, 'brightness', 3),
            SubPolicy(p, 'shearY', 0),
            SubPolicy(p, 'translateX', 1),

            SubPolicy(p, 'sharpness', 5),
            SubPolicy(p, 'invert', 4),
            SubPolicy(p, 'color', 4),
            SubPolicy(p, 'equalize', 8),
            SubPolicy(p, 'rotate', 3),

        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target = super(PolicyDatasetC100, self).__getitem__(index)
        policy_index = torch.zeros(self.policies_len).float()
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
            policy_index[i] = label
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target


class RADatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset):
        super(RADatasetC100, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None
        self.org_dataset=org_dataset
        self.RandAugment = transforms.RandAugment(3,5)

    def __getitem__(self, index):
        sample, target = super(RADatasetC100, self).__getitem__(index)
        new_sample = sample
        new_sample = self.RandAugment(new_sample)
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target


class AADatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset):
        super(AADatasetC100, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None
        self.org_dataset=org_dataset
        self.RandAugment = transforms.RandAugment(3,5)

    def __getitem__(self, index):
        sample, target = super(AADatasetC100, self).__getitem__(index)
        new_sample = sample
        new_sample = self.RandAugment(new_sample)
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target

class PolicyDatasetImageNet(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.2):
        super(PolicyDatasetImageNet, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None
        self.org_dataset=org_dataset
        print("the probability of ImageNet is {}".format(p))
        self.policies = [
            SubPolicy(p, 'posterize', 8),
            SubPolicy(p, 'solarize', 5),
            SubPolicy(p, 'equalize', 8),
            SubPolicy(p, 'posterize', 7),
            SubPolicy(p, 'equalize', 7),

            SubPolicy(p, 'equalize', 4),
            SubPolicy(p, 'solarize', 3),
            SubPolicy(p, 'posterize', 5),
            SubPolicy(p, 'rotate', 3),
            SubPolicy(p, 'equalize', 8),

            SubPolicy(p, 'rotate', 8),
            SubPolicy(p, 'rotate', 9),
            SubPolicy(p, 'equalize', 7),
            SubPolicy(p, 'invert', 4),
            SubPolicy(p, 'color', 4),

            SubPolicy(p, 'rotate', 8),
            SubPolicy(p, 'color', 8),
            SubPolicy(p, 'sharpness', 7),
            SubPolicy(p, 'shearX', 5),
            SubPolicy(p, 'color', 0),

            SubPolicy(p, 'equalize', 7),
            SubPolicy(p, 'solarize', 5),
            SubPolicy(p, 'invert', 4),
            SubPolicy(p, 'color', 4),
            SubPolicy(p, 'equalize', 8)

        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target = super(PolicyDatasetImageNet, self).__getitem__(index)
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target



class ContrastiveDataset(BaseDatasetWrapper):
    def __init__(self, org_dataset, num_negative_samples, mode, ratio,num_classes):
        super().__init__(org_dataset)
        self.num_negative_samples = num_negative_samples
        self.mode = mode
        if isinstance(org_dataset,PolicyDatasetC100):
            org_dataset=org_dataset.org_dataset
        num_classes = num_classes
        num_samples = len(org_dataset)
        labels = org_dataset.targets
        self.cls_positives = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positives[labels[i]].append(i)

        self.cls_negatives = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negatives[i].extend(self.cls_positives[j])

        self.cls_positives = [np.asarray(self.cls_positives[i]) for i in range(num_classes)]
        self.cls_negatives = [np.asarray(self.cls_negatives[i]) for i in range(num_classes)]
        if 0 < ratio < 1:
            n = int(len(self.cls_negatives[0]) * ratio)
            self.cls_negatives = [np.random.permutation(self.cls_negatives[i])[0:n] for i in range(num_classes)]

        self.cls_positives = np.asarray(self.cls_positives)
        self.cls_negatives = np.asarray(self.cls_negatives)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if isinstance(target,int):
            target=[target]*2
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positives[target[0]], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)

        replace = True if self.num_negative_samples > len(self.cls_negatives[target[0]]) else False
        neg_idx = np.random.choice(self.cls_negatives[target[0]], self.num_negative_samples, replace=replace)
        contrast_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        if isinstance(target,list):
            target=target[0]
        else:
            index = torch.LongTensor([index])
            index = index.unsqueeze(0).expand(2, -1)  # 2,1
            contrast_idx = np.tile(contrast_idx[None, ...], (2, 1))
        return sample, target, index, contrast_idx
