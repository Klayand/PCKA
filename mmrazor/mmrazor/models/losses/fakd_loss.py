# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


import numpy as np
import torch, math
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PKDLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(PKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S,), (preds_T,)

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += F.mse_loss(norm_S, norm_T) / 2
        return loss * self.loss_weight


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / 4).softmax(dim=1)
        y_t = (logits_teacher / 4).softmax(dim=1)
        inter_loss = 16 * inter_class_relation(y_s, y_t)
        intra_loss = 16 * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=True))
        # self.res_block = nn.Sequential(
        #     nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=True))

    def forward(self, x):
        y = self.block(x)
        return y
import random

import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .dist_loss import DISTLoss
from .decoupled_kd import  DKDLoss
from .pkd_loss import PKDLoss
from .kl_divergence import  KLDivergence
from .attention import Transformer, WindowAttention
from .layer import ln2d
from functools import partial


import random
import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)))

import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F

# from .losses import DISTLoss, DKDLoss, PKDLoss, KDLoss
from .attention import Transformer, WindowAttention
from .layer import ln2d
from functools import partial


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Copyright (c) OpenMMLab. All rights reserved.


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channel,channel,(1,1),(1,1),(0,0),bias=False),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=False,groups=channel))

    def forward(self, x):
        y = self.block(x)
        return y


def Meta_Encoder(type="conv", window_size=4):
    if type == "conv":
        module = lambda student_channel: nn.Sequential(nn.SiLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1),
                                                                 bias=False),
                                                       nn.GroupNorm(1, student_channel),
                                                       nn.SiLU(inplace=True),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1),
                                                                 bias=False))
    elif type == "resconv":
        class ConvBlock(nn.Module):
            def __init__(self, student_channel):
                super().__init__()
                self.student_channel = student_channel
                self.blcok1 = nn.Sequential(
                    nn.SiLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                              (1, 1),
                              bias=False),
                    nn.GroupNorm(1, student_channel)
                )
                self.blcok2 = nn.Sequential(
                    nn.SiLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                              (1, 1),
                              bias=False),
                    nn.GroupNorm(1, student_channel)
                )

            def forward(self, x):
                x = self.blcok1(x) + x
                x = self.blcok2(x) + x
                return x

        module = lambda student_channel: ConvBlock(student_channel)
    elif type == "transformer":
        attn1 = partial(WindowAttention, window_size=window_size, shifted=True)
        module = lambda student_channel: Transformer(student_channel,
                                                     heads=4, dim_head=64,
                                                     attn=attn1, f=partial(nn.Conv2d, kernel_size=1),
                                                     dropout=0, norm=ln2d)
        return module
    elif type == "mlp":
        class MLP(nn.Module):
            def __init__(self, student_channel, if_norm=False):
                super().__init__()
                self.module = nn.Sequential(
                    nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False))
                self.norm = ln2d(student_channel)
                self.if_norm = if_norm

            def forward(self, x):
                if self.if_norm:
                    return self.norm(self.module(x) + x)
                else:
                    return self.module(x) + x

        module = lambda student_channel: nn.Sequential(
            MLP(student_channel, if_norm=True))  # MLP(student_channel, if_norm=True)
    else:
        raise NotImplementedError
    return module


def lower_power(n):
    exponent = math.floor(math.log2(n))
    return 2 ** exponent





@MODELS.register_module()
class FAKDLoss(nn.Module):
    def __init__(self, teacher_channel, student_channel, ftype="feature_based", teacher_size=None,
                 student_size=None, sampling=8, dirac_ratio=1., weight=1.0,stage=None, apply_warmup=False,
                     loss_type=None, encoder_type=None, apply_model_based_timestep=False, apply_pred_hat_x=False):
        super().__init__()
        self.type = ftype
        assert self.type in ["feature_based", "logit_based"]
        if self.type == "feature_based":
            assert teacher_size is not None and student_size is not None, \
                "For feature-based distillation, FlowAlignModule should " \
                "know the feature map size of teacher intermediate output" \
                " and student intermediate output"
        self.teacher_channel = teacher_channel
        self.student_channel = student_channel
        self.teacher_size = teacher_size
        self.student_size = student_size
        self.time_embedding = student_channel
        self.sampling = sampling
        self.apply_warmup = apply_warmup
        self.apply_model_based_timestep = apply_model_based_timestep
        self.apply_pred_hat_x = apply_pred_hat_x
        self.weight = weight
        print("dirac ratios is:", dirac_ratio, "apply_warmup is:", apply_warmup)
        self.dirac_ratio = 1 - dirac_ratio
        if self.type == "feature_based":
            self.align_loss = PKDLoss()
            if isinstance(teacher_size, tuple):
                teacher_size = teacher_size[0]
            if isinstance(student_size, tuple):
                student_size = student_size[0]
            d = int(teacher_size // student_size)
            self.lowermodule = nn.Sequential(
                nn.BatchNorm2d(self.teacher_channel),
                nn.Conv2d(self.teacher_channel, self.student_channel, (1, 1), (d, d), (0, 0), bias=False))
            self.studentmodule = nn.Identity()  # nn.BatchNorm2d(self.student_channel)
            self.flowembedding = Meta_Encoder(encoder_type if encoder_type is not None else "conv",
                                              window_size=7 if student_size % 7 == 0 else 4)(student_channel)
            self.fc = nn.Identity()
            if self.apply_model_based_timestep:
                self.time_embed = nn.ModuleList([ResBlock(student_channel) for _ in
                                                 range(16)])
            else:
                self.time_embed = nn.Sequential(
                    nn.Linear(self.student_channel, self.student_channel),
                )
        else:
            self.align_loss = DISTLoss()
            self.lowermodule = nn.Identity()
            self.studentmodule = nn.Identity()
            self.flowembedding = Meta_Encoder(encoder_type if encoder_type is not None else "conv",
                                              window_size=7 if student_size % 7 == 0 else 4)(student_channel)
            self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(student_channel, teacher_channel))
            if self.apply_model_based_timestep:
                self.time_embed = nn.ModuleList([ResBlock(student_channel) for _ in
                                                 range(self.sampling)])
            else:
                self.time_embed = nn.Sequential(
                    nn.Linear(self.student_channel, self.student_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.student_channel, self.student_channel))

        if loss_type != None:
            if loss_type == "dkd":
                if teacher_size is not None and teacher_size % 7 == 0:
                    self.align_loss = DKDLoss(alpha=0.5, beta=0.5, warmup=5, temperature=1)
                    print("Apply in ImageNet", teacher_size, student_size, teacher_channel, student_channel)
                else:
                    self.align_loss = DKDLoss(alpha=1.0, beta=2.0, warmup=20, temperature=4)
            elif loss_type == "dist":
                self.align_loss = DISTLoss(tem=4)
            elif loss_type == "pkd":
                self.align_loss = PKDLoss()
            elif loss_type == "mse":
                self.align_loss = nn.MSELoss(reduction="mean")
            elif loss_type == "kl":
                self.align_loss = KDLoss(temperature=4, alpha=0, beta=1)
            else:
                raise NotImplementedError
        required_size=[(300,300),(150,150),(80,80),(45,45)]
        self.pool=nn.AdaptiveAvgPool2d(output_size=required_size[stage])
    def forward(self, student_feature, teacher_feature, inference_sampling=4, **kwargs):
    # def forward(self, preds_S, preds_T, inference_sampling=4, **kwargs):
        student_feature=self.pool(student_feature)
        teacher_feature=self.pool(teacher_feature)
        if self.weight == 0:
            return torch.Tensor([0.]).to(student_feature.device), student_feature

        student_feature = self.studentmodule(student_feature)

        if teacher_feature is not None:
            _len_dirac = int(self.dirac_ratio * teacher_feature.shape[0])
            teacher_feature[:_len_dirac][torch.randperm(_len_dirac, device=student_feature.device)] \
                = teacher_feature[:_len_dirac].clone()
            teacher_feature = teacher_feature.contiguous()

        if self.training:
            """
            Random Sampling Aware
            """
            if isinstance(self.align_loss, DKDLoss):
                align_loss = lambda s, t: self.align_loss(s, t, target=kwargs["target"], epoch=kwargs["epoch"])
            else:
                align_loss = self.align_loss

            if self.type == "feature_based":
                inference_sampling = [1, 2, 4, 8, 16]
            else:
                inference_sampling = [self.sampling]
            inference_sampling = np.random.choice(inference_sampling, 1)[0]
            if self.apply_warmup:
                inference_sampling = min(inference_sampling, int((kwargs["epoch"] + 10) / 10))
                inference_sampling = lower_power(inference_sampling)
            indices = reversed(range(1, inference_sampling + 1))
            x = student_feature
            total_velocity = []
            loss = 0.
            t_output_feature = self.lowermodule(teacher_feature)
            if self.type != "feature_based":
                outputs = []
            for i in indices:
                _weight = self.weight / inference_sampling
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                if not self.apply_model_based_timestep:
                    _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                 self.time_embedding, 1,
                                                                                                 1)
                    embed_x = x + _t_embed
                else:
                    _t_embed = self.time_embed[i - 1](x)
                    embed_x = x + _t_embed
                _velocity = self.flowembedding(embed_x)
                if self.apply_pred_hat_x:
                    _velocity = student_feature - _velocity
                x = x - _velocity / inference_sampling
                total_velocity.append(_velocity)
                if _weight != 0:
                    if self.type == "feature_based":
                        loss += align_loss(self.fc(student_feature) - t_output_feature,
                                           self.fc(_velocity)).mean() * _weight
                    else:
                        output = self.fc(student_feature - _velocity)
                        outputs.append(output)
                        loss += ((align_loss(output, t_output_feature)) + F.cross_entropy(output, kwargs[
                            "target"])).mean() * _weight
            if self.type != "feature_based":
                x = torch.stack(outputs, 0).mean(0)
            # return loss, x
            return loss

        else:
            x = student_feature
            indices = reversed(range(1, inference_sampling + 1))
            if self.type != "feature_based":
                outputs = []
            for i in indices:
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                if not self.apply_model_based_timestep:
                    _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                 self.time_embedding, 1,
                                                                                                 1)
                    embed_x = x + _t_embed
                else:
                    _t_embed = self.time_embed[int(i / inference_sampling * self.sampling) - 1](x)
                    embed_x = x + _t_embed
                _velocity = self.flowembedding(embed_x)
                if self.apply_pred_hat_x:
                    _velocity = student_feature - _velocity
                x = x - _velocity / inference_sampling
                if self.type != "feature_based":
                    output = self.fc(student_feature - _velocity)
                    outputs.append(output)
            if self.type != "feature_based":
                x = torch.stack(outputs, 0).mean(0)
            return torch.Tensor([0.]).to(x.device), x
