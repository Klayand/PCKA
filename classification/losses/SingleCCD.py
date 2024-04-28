import torch
import torch.nn as nn
import torch.nn.functional as F
class CCDLoss(nn.KLDivLoss):
    def __init__(self, temperature, alpha=None, beta=None, p=0.5,reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.p=p
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)
        self.kl_loss=nn.KLDivLoss(reduction="none")

        self.momentum=0.99

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        b1_indices = torch.arange(targets.shape[0]) % 2 == 0
        b2_indices = torch.arange(targets.shape[0]) % 2 != 0
        original_soft_loss = super().forward(torch.log_softmax(student_output[b1_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b1_indices] / self.temperature, dim=1))
        b1=teacher_output[b1_indices]
        b2=teacher_output[b2_indices]
        cosine=F.cosine_similarity(b1,b2)+1
        augmented_soft_loss = self.kl_loss(torch.log_softmax(student_output[b2_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b2_indices] / self.temperature, dim=1))*cosine.unsqueeze(-1)
        augmented_soft_loss = augmented_soft_loss.sum(-1).mean()
        soft_loss=(original_soft_loss+augmented_soft_loss)/2
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
