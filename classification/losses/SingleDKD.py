import torch
import torch.nn as nn
import torch.nn.functional as F
class DoupleKDLoss(nn.KLDivLoss):
    def __init__(self, temperature, weight=[1,1,1,1], p=0.5,reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.weight = weight
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.p=p
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)
        self.kl_loss=nn.KLDivLoss(reduction="none")

        self.momentum=0.99

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        b1_indices = torch.arange(targets.shape[0]) % 2 == 0
        b2_indices = torch.arange(targets.shape[0]) % 2 != 0
        original_soft_loss = super().forward(torch.log_softmax(student_output[b1_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b1_indices] / self.temperature, dim=1))*self.weight[2]
        augmented_soft_loss = super().forward(torch.log_softmax(student_output[b2_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b2_indices] / self.temperature, dim=1))*self.weight[3]
        augmented_soft_loss = augmented_soft_loss.sum(-1).mean()
        soft_loss=(original_soft_loss+augmented_soft_loss)/2
        original_hard_loss = self.cross_entropy_loss(student_output[b1_indices], targets[b1_indices]) * self.weight[0]
        augmented_hard_loss = self.cross_entropy_loss(student_output[b2_indices], targets[b2_indices]) * self.weight[1]
        hard_loss = (original_hard_loss+augmented_hard_loss)/2

        return hard_loss + (self.temperature ** 2) * soft_loss
