import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(a, b, eps=1e-5):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-5):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2, tem=4):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem**2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem**2) * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd