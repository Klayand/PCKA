# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


@MODELS.register_module()
class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, loss_weight=1.0):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss


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
class ChannelWiseDivergence_patch(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, loss_weight=1.0):
        super(ChannelWiseDivergence_patch, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[1] == preds_T.shape[1]
        C = preds_S.shape[1]

        t_feature_cropped, s_feature_cropped = patches(preds_T, preds_S)

        loss_cwd = 0
        for i in range(C):
            softmax_pred_T = F.softmax(t_feature_cropped[i] / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(t_feature_cropped[i] / self.tau) -
                             softmax_pred_T *
                             logsoftmax(s_feature_cropped[i] / self.tau)) * (
                                 self.tau**2)

            loss_cwd += self.loss_weight * loss

        return loss_cwd / C