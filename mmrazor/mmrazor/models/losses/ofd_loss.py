# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


@MODELS.register_module()
class OFDLoss(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation
    https://sites.google.com/view/byeongho-heo/overhaul.

    The partial L2loss, only calculating loss when
    `out_s > out_t` or `out_t > 0`.

    Args:
        loss_weight (float, optional): loss weight. Defaults to 1.0.
        mul_factor (float, optional): multiply factor. Defaults to 1000.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 mul_factor: float = 1000.) -> None:
        super(OFDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.mul_factor = mul_factor

    def forward_train(self, s_feature: torch.Tensor,
                      t_feature: torch.Tensor) -> torch.Tensor:
        """forward func for training.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        bsz = s_feature.shape[0]
        loss = torch.nn.functional.mse_loss(
            s_feature, t_feature, reduction='none')
        loss = loss * ((s_feature > t_feature) | (t_feature > 0)).float()
        return loss.sum() / bsz / self.mul_factor

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """forward func.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        return self.loss_weight * self.forward_train(s_feature, t_feature)


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
class OFDLoss_patch(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation
    https://sites.google.com/view/byeongho-heo/overhaul.

    The partial L2loss, only calculating loss when
    `out_s > out_t` or `out_t > 0`.

    Args:
        loss_weight (float, optional): loss weight. Defaults to 1.0.
        mul_factor (float, optional): multiply factor. Defaults to 1000.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 mul_factor: float = 1000.) -> None:
        super(OFDLoss_patch, self).__init__()
        self.loss_weight = loss_weight
        self.mul_factor = mul_factor

    def forward_train(self, s_feature: torch.Tensor,
                      t_feature: torch.Tensor) -> torch.Tensor:
        """forward func for training.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        assert s_feature.shape[1] == t_feature.shape[1]
        C = s_feature.shape[1]

        t_feature_cropped, s_feature_cropped = patches(t_feature, s_feature)

        loss_ofd = 0

        for i in range(C):
            loss = torch.nn.functional.mse_loss(
                s_feature_cropped[i], t_feature_cropped[i], reduction='none')
            loss_ofd += (loss * ((s_feature_cropped[i] > t_feature_cropped[i]) | (t_feature_cropped[i] > 0)).float()).sum()
        return loss_ofd / C / self.mul_factor

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """forward func.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        return self.loss_weight * self.forward_train(s_feature, t_feature)
