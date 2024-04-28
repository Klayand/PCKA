import torch
import torch.nn as nn
import torch.nn.functional as F
class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)


    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return self.beta * (self.temperature ** 2) * soft_loss
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
