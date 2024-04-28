# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


@MODELS.register_module()
class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return self.loss_weight * loss


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
class KLDivergence_patch(nn.Module):
    """
        No temperature
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence_patch, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[1] == preds_T.shape[1]
        C = preds_S.shape[1]

        preds_T_cropped, preds_S_cropped = patches(preds_T, preds_S)

        loss_kl = 0
        for i in range(C):
            softmax_pred_T = F.softmax(preds_T_cropped[i].squeeze() / self.tau, dim=1)
            logsoftmax_preds_S = F.log_softmax(preds_S_cropped[i].squeeze() / self.tau, dim=1)
            loss_kl += (self.tau**2) * F.kl_div(
                logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)

        return 1 / C * self.loss_weight * loss_kl
