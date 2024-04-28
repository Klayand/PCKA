# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


@MODELS.register_module()
class ATLoss(nn.Module):
    """"Paying More Attention to Attention: Improving the Performance of
    Convolutional Neural Networks via Attention Transfer" Conference paper at
    ICLR2017 https://openreview.net/forum?id=Sks9_ajex.

    https://github.com/szagoruyko/attention-transfer/blob/master/utils.py

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """"Forward function for ATLoss."""
        loss = (self.calc_attention_matrix(s_feature) -
                self.calc_attention_matrix(t_feature)).pow(2).mean()
        return self.loss_weight * loss

    def calc_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def div_sixteen_mul(v):
    v = int(v)
    m = v % 16
    return int(v // 16 * 16) + int(m > 0) * 16


def patches(teacher_tensor, student_tensor):
    teacher_tensor, student_tensor = teacher_tensor.permute(1, 0, 2, 3), student_tensor.permute(1, 0, 2, 3)

    assert (
        div_sixteen_mul(teacher_tensor.shape[2]) == div_sixteen_mul(student_tensor.shape[2])
        and
        div_sixteen_mul(teacher_tensor.shape[3]) == div_sixteen_mul(student_tensor.shape[3])
    )

    h_p, w_p = div_sixteen_mul(teacher_tensor.shape[2]), div_sixteen_mul(teacher_tensor.shape[3])

    new_teacher_tensor = F.interpolate(teacher_tensor, [h_p, w_p], mode='bilinear')
    new_student_tensor = F.interpolate(student_tensor, [h_p, w_p], mode='bilinear')

    teacher_patches = rearrange(new_teacher_tensor, 'c b (u h) (v w) -> c (u v) (b h w)', h=16, w=16).contiguous()
    student_patches = rearrange(new_student_tensor, 'c b (u h) (v w) -> c (u v) (b h w)', h=16, w=16).contiguous()

    return teacher_patches, student_patches


@MODELS.register_module()
class ATLoss_patch(nn.Module):
    """
    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """"Forward function for ATLoss."""

        assert s_feature.shape[1] == t_feature.shape[1]
        C = s_feature.shape[1]

        t_feature_cropped, s_feature_cropped = patches(t_feature, s_feature)

        loss_at = (self.calc_attention_matrix(s_feature_cropped) -
                   self.calc_attention_matrix(t_feature_cropped)).pow(2).mean()

        return 1 / C * self.loss_weight * loss_at

    def calc_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        return F.normalize(x.sum(dim=0))


@MODELS.register_module()
class ATLoss_patch_linear(nn.Module):
    """
    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """"Forward function for ATLoss."""

        assert s_feature.shape[1] == t_feature.shape[1]
        C = s_feature.shape[1]

        t_feature_cropped, s_feature_cropped = patches(t_feature, s_feature)

        loss_at = (self.calc_attention_matrix(s_feature_cropped) -
                   self.calc_attention_matrix(t_feature_cropped)).pow(2).mean()

        return 1 / C * self.loss_weight * loss_at

    def calc_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        return F.normalize(x).mean(dim=0)


@MODELS.register_module()
class ATLoss_patch_mmd(nn.Module):
    """
    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """"Forward function for ATLoss."""

        assert s_feature.shape[1] == t_feature.shape[1]
        C = s_feature.shape[1]

        t_feature_cropped, s_feature_cropped = patches(t_feature, s_feature)

        loss_at = 0
        for i in range(C):
            gram_s = self.calc_attention_gram_matrix(s_feature_cropped[i].squeeze())
            gram_t = self.calc_attention_gram_matrix(t_feature_cropped[i].squeeze())
            loss_at += (
                (gram_s - gram_t).pow(2).mean() - self.alpha * (gram_s.pow(2) + gram_t.pow(2)).mean()
            )

        return 1 / C * self.loss_weight * loss_at

    def calc_attention_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        return F.normalize(torch.matmul(x, x.t()))
