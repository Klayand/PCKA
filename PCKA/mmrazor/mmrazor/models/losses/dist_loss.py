# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@MODELS.register_module()
class DISTLoss(nn.Module):

    def __init__(
        self,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, logits_S, logits_T: torch.Tensor):
        if self.teacher_detach:
            logits_T = logits_T.detach()
        y_s = (logits_S / self.tau).softmax(dim=1)
        y_t = (logits_T / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return kd_loss * self.loss_weight


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
class DISTLoss_patch(nn.Module):

    def __init__(
        self,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss_patch, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, logits_S, logits_T: torch.Tensor):

        assert logits_S.shape[1] == logits_T.shape[1]
        C = logits_S.shape[1]

        preds_T_cropped, preds_S_cropped = patches(logits_T, logits_S)

        kd_loss = 0
        for i in range(C):
            y_s = (preds_S_cropped[i] / self.tau).softmax(dim=1)
            y_t = (preds_T_cropped[i] / self.tau).softmax(dim=1)
            inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
            intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
            kd_loss += self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return 1 / C * kd_loss * self.loss_weight
