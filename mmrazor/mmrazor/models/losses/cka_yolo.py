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
    B = teacher_tensor.shape[0]
    teacher_tensor, student_tensor = teacher_tensor.permute(1, 0, 2, 3), student_tensor.permute(1, 0, 2, 3)

    assert (
        div_sixteen_mul(teacher_tensor.shape[2]) == div_sixteen_mul(student_tensor.shape[2])
        and
        div_sixteen_mul(teacher_tensor.shape[3]) == div_sixteen_mul(student_tensor.shape[3])
    )

    h_p, w_p = div_sixteen_mul(teacher_tensor.shape[2]), div_sixteen_mul(teacher_tensor.shape[3])

    new_teacher_tensor = F.interpolate(teacher_tensor, [h_p, w_p], mode='bilinear')
    new_student_tensor = F.interpolate(student_tensor, [h_p, w_p], mode='bilinear')

    # accelerate
    squeeze_module = nn.Conv2d(B, 1, kernel_size=1).to(teacher_tensor.device)

    new_student_tensor = squeeze_module(new_student_tensor)
    new_teacher_tensor = squeeze_module(new_teacher_tensor)

    teacher_patches = rearrange(new_teacher_tensor, 'c b (u h) (v w) -> c (u v) (b h w)', h=16, w=16).contiguous()
    student_patches = rearrange(new_student_tensor, 'c b (u h) (v w) -> c (u v) (b h w)', h=16, w=16).contiguous()

    # print(teacher_patches.shape)

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

    # print(gram_X.shape, gram_Y.shape)
    # print(torch.matmul(gram_X, gram_X.t()).shape)

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

    # print("teacher:", teacher_logits.shape)
    # print("student:", student_logits.shape)
    # get a similarity score between teacher and student
    similarity_martix = CenterKernelAlignment(teacher_logits, student_logits, with_l2_norm)

    # maximize the likelihood of it
    return -similarity_martix

@MODELS.register_module()
class GaussianKernel(nn.Module):
    def __init__(self, size):
        super(GaussianKernel, self).__init__()
        self.size = size
        self.mu = torch.zeros(size, size)
        # self.sigma = torch.ones(size, size) / (size ** 2)
        self.sigma = torch.sqrt(torch.tensor(1.0 / (size ** 2)))

    def forward(self, teacher, student, down_channel):

        B, C, H, W = teacher.shape

        gaussian_filter = nn.Conv2d(
            in_channels=C,
            out_channels=down_channel,
            kernel_size=self.size,
            stride=2,
            padding=0,
            bias=False
        ).to(student.device)

        noise = torch.randn(C, down_channel, self.size, self.size) * self.sigma
        kernel = self.mu + noise

        # if H > self.size:
        gaussian_filter.weight.data = kernel.view(down_channel, C, self.size, self.size).to(student.device) # [out, in, kernel, kernel]
        teacher = gaussian_filter(teacher)
        student = gaussian_filter(student)

        # print(teacher.shape)
        # output_downsampled = F.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)

        return teacher, student


@MODELS.register_module()
class CKA_Yolo(nn.Module):
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
        super(CKA_Yolo, self).__init__()
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.down_sample = GaussianKernel(size=4)

        self.down_channel = 64

    def forward(self, student_logits, teacher_logits):
        # print("student: ", student_logits.shape)
        # print("teacher: ", teacher_logits.shape)

        assert student_logits.shape[1] == teacher_logits.shape[1]
        C, H = student_logits.shape[1], student_logits.shape[2]

        # print("student: ", student_logits.shape)
        # print("teacher: ", teacher_logits.shape)

        if H > 20:
            teacher_logits, student_logits = self.down_sample.forward(teacher_logits, student_logits, self.down_channel)

        # print("student down: ", student_logits.shape)
        # print("teacher down: ", teacher_logits.shape)

        teacher_logits_cropped, student_logits_cropped = patches(teacher_logits, student_logits)
        # print(teacher_logits_cropped.shape)

        # print("teacher patches:", teacher_logits_cropped.shape)
        # print("student_patches:", student_logits_cropped.shape)

        # import time

        # time1 = time.time()
        loss_cka = 0

        for i in range(self.down_channel):
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
        # print(C)
        # time2 = time.time()
        # print(time2 - time1)

        total_loss = 1/C * loss_cka

        return total_loss