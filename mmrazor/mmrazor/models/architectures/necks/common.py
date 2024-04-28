import numpy as np
import math
from mmrazor.models.architectures.necks.attention import Transformer, WindowAttention
from mmrazor.models.architectures.necks.layer import ln2d, DropPath
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

LayerNorm = ln2d


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,last_norm=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.last_norm = last_norm

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            gamma = self.gamma.data
            # gamma[gamma.abs()>2] = 2
            x = gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if not self.last_norm:
            input = self.norm(input)
        x = input + self.drop_path(x)
        if self.last_norm:
            x = self.norm(x)
        return x

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
        feat = (feat - mean) / (std + 1e-8)
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
            loss += (F.mse_loss(norm_S, norm_T))/ 2
        return loss * self.loss_weight


class IMSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(IMSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

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

            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += F.mse_loss(pred_S, pred_T) / 2
        return loss * self.loss_weight



def hyp_split(loss_type: str):
    hyp = loss_type.split("_")[1:]
    hyp_dict = {}
    for i, j in zip(hyp[::2], hyp[1::2]):
        hyp_dict[i] = float(j)
    return hyp_dict


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
        self.block = nn.Sequential(nn.Conv2d(channel, channel, (1, 1), (1, 1), (0, 0), bias=False),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=False, groups=channel))

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
    elif type == "conv_mobilenet":
        module = lambda student_channel: nn.Sequential(nn.ReLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1),
                                                                 (0, 0), bias=False),
                                                       nn.GroupNorm(1, student_channel),
                                                       nn.ReLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1), groups=student_channel, bias=False))
    elif type == "conv_detection":
        class ResidualModule(nn.Module):
 
            def __init__(self,module):
                super().__init__()
                self.module = module

            def forward(self,x):
                x = self.module(x) + x
                return x
                
        module = lambda student_channel: nn.Sequential(ResidualModule(nn.Sequential(nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False),
                                                                                    ln2d(student_channel),
                                                                                    nn.ReLU(inplace=True),
                                                                                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1), (1, 1), bias=False),
                                                                                    ln2d(student_channel))),
                                                       ResidualModule(nn.Sequential(nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False),
                                                                                    ln2d(student_channel),
                                                                                    nn.ReLU(inplace=True),
                                                                                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1), (1, 1), bias=False),
                                                                                    ln2d(student_channel))),
                                                       nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False))
        return module
    
    elif type == "convnext_detection":
                
        module = lambda student_channel: nn.Sequential(Block(student_channel),Block(student_channel))
        return module

    elif type == "convnext_detection_without_last_norm":
                
        module = lambda student_channel: nn.Sequential(Block(student_channel,last_norm=False),Block(student_channel,last_norm=False))
        return module
    
    elif type == "transformer_detection":
        attn1 = partial(WindowAttention, window_size=7, shifted=True)
        attn2 = partial(WindowAttention, window_size=7, shifted=False)
        
        module = lambda student_channel: nn.Sequential(
            Transformer(student_channel,
            heads=4, dim_head=student_channel//8, dim_mlp=student_channel,
            attn=attn1, f=partial(nn.Conv2d, kernel_size=1),
            dropout=0.05, norm=ln2d),
            Transformer(student_channel,
            heads=4, dim_head=student_channel//8, dim_mlp=student_channel,
            attn=attn2, f=partial(nn.Conv2d, kernel_size=1),
            dropout=0.05, norm=ln2d))
        return module

    elif type == "transformer":
        attn1 = partial(WindowAttention, window_size=window_size, shifted=True)
        module = lambda student_channel: Transformer(student_channel,
                                                     heads=4, dim_head=32, dim_mlp=256,
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
            MLP(student_channel, if_norm=True))  # ,MLP(student_channel, if_norm=True)
    else:
        raise NotImplementedError
    return module


def lower_power(n):
    exponent = math.floor(math.log2(n))
    return 2 ** exponent


class FlowAlignModule(nn.Module):
    def __init__(self, teacher_channel, student_channel, type="feature_based", teacher_size=None,
                 student_size=None, sampling=8, dirac_ratio=1., weight=1.0, apply_warmup=False, encoder_type=None,
                 apply_model_based_timestep=False, apply_pred_hat_x=False):
        super().__init__()
        self.type = type
        assert self.type in ["feature_based", "logit_based"]
        if self.type == "feature_based":
            if teacher_size is None and student_size is None:
                teacher_size = student_size = 1  # only for mmdet
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
        self.iter_number = 0
        if self.weight == 0:
            return
        print("sampling is:", sampling, "\tdirac ratios is:", dirac_ratio,
              "\tapply_warmup is:", apply_warmup, "\tweight is:", weight,
              "\tencoder type is:", encoder_type, "\tapply model based timestep is:",
              apply_model_based_timestep)
        self.dirac_ratio = 1 - dirac_ratio
        if self.type == "feature_based":
            self.align_loss = PKDLoss()
            if isinstance(teacher_size, tuple):
                teacher_size = teacher_size[0]
            if isinstance(student_size, tuple):
                student_size = student_size[0]
            d = int(teacher_size // student_size)
            if self.student_channel != self.teacher_channel:
                self.lowermodule = nn.Sequential(
                    nn.BatchNorm2d(self.teacher_channel),
                    nn.Conv2d(self.teacher_channel, self.student_channel, (1, 1), (d, d), (0, 0), bias=False))
            else:
                self.lowermodule = nn.Identity()
            self.studentmodule = nn.Identity()  # nn.BatchNorm2d(self.student_channel)
            self.flowembedding = Meta_Encoder(encoder_type if encoder_type is not None else "conv",
                                              window_size=7 if student_size % 7 == 0 else 4)(student_channel)
            if self.apply_model_based_timestep:
                self.time_embed = nn.ModuleList([ResBlock(student_channel) for _ in
                                                 range(16)])
            else:
                self.time_embed = nn.Sequential(
                    nn.Linear(self.student_channel, self.student_channel),
                )
        else:
            raise NotImplementedError("In detection task, logit-based fakd is not implemented")
        
        self.fc = nn.ModuleList([nn.Identity() for i in range(8)])

    def forward(self, student_feature, teacher_feature, inference_sampling=4, **kwargs):
        if self.weight == 0:
            return torch.Tensor([0.]).to(student_feature.device), student_feature

        student_feature = self.studentmodule(student_feature)

        if teacher_feature is not None:
            _len_dirac = int(self.dirac_ratio * teacher_feature.shape[0])
            teacher_feature[:_len_dirac][torch.randperm(_len_dirac, device=student_feature.device)] \
                = teacher_feature[:_len_dirac].clone()
            teacher_feature = teacher_feature.contiguous()
        total_velocity = []
        if self.training:
            """
            Random Sampling Aware
            """
            align_loss = self.align_loss
            inference_sampling = [4]
            inference_sampling = np.random.choice(inference_sampling, 1)[0]
            if self.apply_warmup:
                inference_sampling = min(inference_sampling, int((kwargs["epoch"] + 10) / 10))
                inference_sampling = lower_power(inference_sampling)
            indices = reversed(range(1, inference_sampling + 1))
            x = student_feature
            loss = 0.
            t_output_feature = self.lowermodule(teacher_feature)
            
            n_weight = 1 ** torch.Tensor(list(deepcopy(indices))).to(x.device)
            n_weight = n_weight * self.weight / n_weight.sum()
            for i in indices:
                _weight = n_weight[-i]
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
                hat_x = self.fc[-i](student_feature - _velocity)
                total_velocity.append(hat_x)
                if _weight != 0:
                    loss += align_loss(self.fc[-i](student_feature - _velocity), t_output_feature).mean() * _weight
            x = torch.stack(total_velocity,0).mean(0)
            return loss, x

        else:
            x = student_feature
            indices = reversed(range(1, inference_sampling + 1))
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
                hat_x = self.fc[-i](student_feature - _velocity)
                total_velocity.append(hat_x)
            return torch.Tensor([0.]).to(x.device), torch.stack(total_velocity,0).mean(0)
