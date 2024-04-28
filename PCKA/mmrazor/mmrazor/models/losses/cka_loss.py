import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat

from mmrazor.registry import MODELS


def random_crop_tensor(teacher_tensor, student_tensor, crop_size=(32, 32)):
    original_size = teacher_tensor.size()

    if teacher_tensor.size()[2] < 32 or teacher_tensor.size()[3] < 32:
        teacher_tensor = teacher_tensor.permute(1, 0, 2, 3)
        student_tensor = student_tensor.permute(1, 0, 2, 3)

    else:
        top = torch.randint(0, original_size[2] - crop_size[0] + 1, (1,))
        left = torch.randint(0, original_size[3] - crop_size[1] + 1, (1,))
        teacher_cropped_tensor = teacher_tensor[:, :, top:top + crop_size[0],
                                 left:left + crop_size[1]]
        student_cropped_tensor = student_tensor[:, :, top:top + crop_size[0],
                                 left:left + crop_size[1]]

        teacher_tensor = teacher_cropped_tensor.permute(1, 0, 2, 3)
        student_tensor = student_cropped_tensor.permute(1, 0, 2, 3)

    return teacher_tensor, student_tensor


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


def patches_batch(teacher_tensor, student_tensor):
    teacher_tensor, student_tensor = teacher_tensor.permute(1, 0, 2, 3), student_tensor.permute(1, 0, 2, 3)

    assert (
        div_sixteen_mul(teacher_tensor.shape[2]) == div_sixteen_mul(student_tensor.shape[2])
        and
        div_sixteen_mul(teacher_tensor.shape[3]) == div_sixteen_mul(student_tensor.shape[3])
    )

    h_p, w_p = div_sixteen_mul(teacher_tensor.shape[2]), div_sixteen_mul(teacher_tensor.shape[3])

    new_teacher_tensor = F.interpolate(teacher_tensor, [h_p, w_p], mode='bilinear')
    new_student_tensor = F.interpolate(student_tensor, [h_p, w_p], mode='bilinear')

    teacher_patches = rearrange(new_teacher_tensor, 'c b (u h) (v w) -> b (u v) (c h w)', h=16, w=16).contiguous()
    student_patches = rearrange(new_student_tensor, 'c b (u h) (v w) -> b (u v) (c h w)', h=16, w=16).contiguous()

    return teacher_patches, student_patches


def patches_hw(teacher_tensor, student_tensor):
    teacher_tensor, student_tensor = teacher_tensor.permute(1, 0, 2, 3), student_tensor.permute(1, 0, 2, 3)

    assert (
        div_sixteen_mul(teacher_tensor.shape[2]) == div_sixteen_mul(student_tensor.shape[2])
        and
        div_sixteen_mul(teacher_tensor.shape[3]) == div_sixteen_mul(student_tensor.shape[3])
    )

    h_p, w_p = div_sixteen_mul(teacher_tensor.shape[2]), div_sixteen_mul(teacher_tensor.shape[3])

    new_teacher_tensor = F.interpolate(teacher_tensor, [h_p, w_p], mode='bilinear')
    new_student_tensor = F.interpolate(student_tensor, [h_p, w_p], mode='bilinear')

    teacher_patches = rearrange(new_teacher_tensor, 'c b (u h) (v w) -> (h w) (u v) (b c)', h=16, w=16).contiguous()
    student_patches = rearrange(new_student_tensor, 'c b (u h) (v w) -> (h w) (u v) (b c)', h=16, w=16).contiguous()

    return teacher_patches, student_patches

def CenterKernelAlignment(X, Y, with_l2_norm):
    """Compute the CKA similarity betweem samples"""
    # Compute Gram matrix
    gram_X = torch.matmul(X, X.t())
    gram_Y = torch.matmul(Y, Y.t())

    # l2 norm or not
    if with_l2_norm:
        gram_X = gram_X / torch.sqrt(torch.diag(gram_X)[:, None])
        gram_Y = gram_Y / torch.sqrt(torch.diag(gram_Y)[:, None])

    # compute cka
    cka = torch.trace(torch.matmul(gram_X, gram_Y.t())) / torch.sqrt(
        torch.trace(torch.matmul(gram_X, gram_X.t())) * torch.trace(torch.matmul(gram_Y, gram_Y.t()))
    )

    return cka


def cka_loss(teacher_logits, student_logits, with_l2_norm):
    """Compute the CKA similarity between samples
    CKA computes similarity between batches
    input: (N, P) ----> output: (N, N) similarity matrix
    """
    N_t = teacher_logits.shape[0]
    N_s = student_logits.shape[0]
    assert N_s == N_t  # when use cka, you need to make sure N the same

    # get a similarity score between teacher and student
    similarity_martix = CenterKernelAlignment(teacher_logits, student_logits, with_l2_norm)

    # maximize the likelihood of it
    return -similarity_martix


@MODELS.register_module()
class CenterKernelAlignmentRKD(nn.Module):
    """Center Kernel Alignment for Relational Knowledge Distillation"""

    def __init__(
            self,
            ce_weight=1.0,
            intra_weight=15,
            inter_weight=None,
            combined_KD=False,
            temperature=None,
            with_l2_norm=False,
    ):
        super(CenterKernelAlignmentRKD, self).__init__()
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD

    def forward(self, student_logits, teacher_logits):

        assert student_logits.shape[1] == teacher_logits.shape[1]
        C = student_logits.shape[1]

        teacher_logits_cropped, student_logits_cropped = patches(teacher_logits, student_logits)


        # student_logits_cropped = student_logits_cropped.reshape(student_logits.shape[0], -1)
        # teacher_logits_cropped = teacher_logits_cropped.reshape(teacher_logits.shape[0], -1)
        #
        # student_mean = student_logits_cropped.mean(dim=-1, keepdim=True)
        # student_std = student_logits_cropped.std(dim=-1, keepdim=True)
        # student_logits_cropped = (student_logits_cropped - student_mean) / (student_std + 1e-6)
        #
        # teacher_mean = teacher_logits_cropped.mean(dim=-1, keepdim=True)
        # teacher_std = teacher_logits_cropped.std(dim=-1, keepdim=True)
        # teacher_logits_cropped = (teacher_logits_cropped - teacher_mean) / (teacher_std + 1e-6)

        loss_cka = 0
        for i in range(C):
            loss_cka_intra = self.intra_weight * cka_loss(
                teacher_logits=teacher_logits_cropped[i].squeeze(), student_logits=student_logits_cropped[i].squeeze(),
                with_l2_norm=self.with_l2_norm
            )

            loss_cka += loss_cka_intra

            if self.inter_weight:
                loss_cka_inter = self.intra_weight * cka_loss(
                    teacher_logits=teacher_logits_cropped[i].squeeze().transpose(0, 1),
                    student_logits=student_logits_cropped[i].squeeze().transpose(0, 1),
                    with_l2_norm=self.with_l2_norm
                )

                loss_cka += loss_cka_inter

        total_loss = 1/C * loss_cka

        if self.combined_KD:
            from .kl_divergence import KLDivergence

            kd_loss = KLDivergence(tau=self.temperature)
            loss_kd = kd_loss(student_logits_cropped, teacher_logits_cropped)
            total_loss += loss_kd

        return total_loss


@MODELS.register_module()
class CenterKernelAlignmentRKD_batch(nn.Module):
    """Center Kernel Alignment for Relational Knowledge Distillation"""

    def __init__(
            self,
            ce_weight=1.0,
            intra_weight=15,
            inter_weight=None,
            combined_KD=False,
            temperature=None,
            with_l2_norm=False,
    ):
        super(CenterKernelAlignmentRKD_batch, self).__init__()
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD

    def forward(self, student_logits, teacher_logits):

        assert student_logits.shape[1] == teacher_logits.shape[1]
        B = student_logits.shape[0]

        teacher_logits_cropped, student_logits_cropped = patches_batch(teacher_logits, student_logits)

        loss_cka = 0
        for i in range(B):
            loss_cka_intra = self.intra_weight * cka_loss(
                teacher_logits=teacher_logits_cropped[i].squeeze(), student_logits=student_logits_cropped[i].squeeze(),
                with_l2_norm=self.with_l2_norm
            )

            loss_cka += loss_cka_intra

            if self.inter_weight:
                loss_cka_inter = self.intra_weight * cka_loss(
                    teacher_logits=teacher_logits_cropped[i].squeeze().transpose(0, 1),
                    student_logits=student_logits_cropped[i].squeeze().transpose(0, 1),
                    with_l2_norm=self.with_l2_norm
                )

                loss_cka += loss_cka_inter

        total_loss = 1/B * loss_cka

        if self.combined_KD:
            from .kl_divergence import KLDivergence

            kd_loss = KLDivergence(tau=self.temperature)
            loss_kd = kd_loss(student_logits_cropped, teacher_logits_cropped)
            total_loss += loss_kd

        return total_loss


@MODELS.register_module()
class CenterKernelAlignmentRKD_hw(nn.Module):
    """Center Kernel Alignment for Relational Knowledge Distillation"""

    def __init__(
            self,
            ce_weight=1.0,
            intra_weight=15,
            inter_weight=None,
            combined_KD=False,
            temperature=None,
            with_l2_norm=False,
    ):
        super(CenterKernelAlignmentRKD_hw, self).__init__()
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD

    def forward(self, student_logits, teacher_logits):

        assert student_logits.shape[1] == teacher_logits.shape[1]

        teacher_logits_cropped, student_logits_cropped = patches_hw(teacher_logits, student_logits)

        hw = teacher_logits_cropped.shape[0]

        loss_cka = 0
        for i in range(hw):
            loss_cka_intra = self.intra_weight * cka_loss(
                teacher_logits=teacher_logits_cropped[i].squeeze(), student_logits=student_logits_cropped[i].squeeze(),
                with_l2_norm=self.with_l2_norm
            )

            loss_cka += loss_cka_intra

            if self.inter_weight:
                loss_cka_inter = self.intra_weight * cka_loss(
                    teacher_logits=teacher_logits_cropped[i].squeeze().transpose(0, 1),
                    student_logits=student_logits_cropped[i].squeeze().transpose(0, 1),
                    with_l2_norm=self.with_l2_norm
                )

                loss_cka += loss_cka_inter

        total_loss = 1/hw * loss_cka

        if self.combined_KD:
            from .kl_divergence import KLDivergence

            kd_loss = KLDivergence(tau=self.temperature)
            loss_kd = kd_loss(student_logits_cropped, teacher_logits_cropped)
            total_loss += loss_kd

        return total_loss


@MODELS.register_module()
class CenterKernelAlignmentSeg(nn.Module):
    """Center Kernel Alignment for Relational Knowledge Distillation"""

    def __init__(
            self,
            ce_weight=1.0,
            intra_weight=15,
            inter_weight=None,
            combined_KD=False,
            temperature=None,
            with_l2_norm=True,
            soften=False,
    ):
        super(CenterKernelAlignmentSeg, self).__init__()
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.soften = soften

    def forward(self, student_logits, teacher_logits):

        student_logits_cropped = student_logits.reshape(student_logits.shape[0], -1)
        teacher_logits_cropped = teacher_logits.reshape(teacher_logits.shape[0], -1)

        if self.soften:
            logits_teacher = (teacher_logits_cropped / self.temperature).softmax(dim=1)
            logits_student = (student_logits_cropped / self.temperature).softmax(dim=1)

            loss_cka_intra = self.intra_weight * cka_loss(
                teacher_logits=logits_teacher, student_logits=logits_student, with_l2_norm=self.with_l2_norm
            )

        else:
            loss_cka_intra = self.intra_weight * cka_loss(
                teacher_logits=teacher_logits_cropped, student_logits=student_logits_cropped,
                with_l2_norm=self.with_l2_norm
            )

        loss_cka = loss_cka_intra

        if self.inter_weight:
            loss_cka_inter = self.intra_weight * cka_loss(
                teacher_logits=teacher_logits_cropped.transpose(0, 1),
                student_logits=student_logits_cropped.transpose(0, 1),
                with_l2_norm=self.with_l2_norm
            )

            loss_cka = loss_cka_intra + loss_cka_inter

        total_loss = loss_cka

        if self.combined_KD:
            from .kl_divergence import KLDivergence

            kd_loss = KLDivergence(tau=self.temperature)
            loss_kd = kd_loss(student_logits_cropped, teacher_logits_cropped)
            total_loss += loss_kd

        return total_loss