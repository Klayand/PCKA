import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity

class SPKDLoss(nn.Module):
    """
    "Similarity-Preserving Knowledge Distillation"
    """
    def __init__(self, reduction, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.kl_loss=nn.KLDivLoss(reduction=reduction)
        self.cross_entropy_loss=nn.CrossEntropyLoss()
        self.temperature=4

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_feature_map, teacher_feature_map, student_output, teacher_output, targets=None, *args, **kwargs):
        teacher_outputs = teacher_feature_map
        student_outputs = student_feature_map
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        spkd_loss=spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss
        soft_loss = self.kl_loss(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return  hard_loss + spkd_loss + (self.temperature**2)*soft_loss
