import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity

class SSKDLoss(nn.Module):
    """
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
    """
    def __init__(self, kl_temp=4.0, ss_temp=0.5, tf_temp=4.0, ss_ratio=0.75, tf_ratio=1.0,loss_weights=None, reduction='batchmean',
                  **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 0.9, 10.0, 2.7] if loss_weights is None else loss_weights
        self.kl_temp = kl_temp
        self.ss_temp = ss_temp
        self.tf_temp = tf_temp
        self.ss_ratio = ss_ratio
        self.tf_ratio = tf_ratio
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
    @staticmethod
    def compute_cosine_similarities(ss_module_outputs, normal_indices, aug_indices,
                                    three_forth_batch_size, one_forth_batch_size):
        normal_feat = ss_module_outputs[normal_indices]
        aug_feat = ss_module_outputs[aug_indices]
        normal_feat = normal_feat.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_feat = aug_feat.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        return cosine_similarity(aug_feat, normal_feat, dim=1)

    def forward(self,student_ss_module_outputs,teacher_ss_module_outputs , student_linear_outputs,teacher_linear_outputs, targets, *args, **kwargs):
        device = student_linear_outputs.device
        batch_size = student_linear_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = (torch.arange(batch_size) % 4 == 0)
        aug_indices = (torch.arange(batch_size) % 4 != 0)
        ce_loss = self.cross_entropy_loss(student_linear_outputs[normal_indices], targets)
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs[normal_indices] / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs[normal_indices] / self.kl_temp, dim=1))
        kl_loss *= (self.kl_temp ** 2)

        # error level ranking
        aug_knowledges = torch.softmax(teacher_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        aug_targets = targets.unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(aug_knowledges, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.tf_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_tf = torch.sort(indices)[0]
        s_cos_similarities = self.compute_cosine_similarities(student_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = self.compute_cosine_similarities(teacher_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = t_cos_similarities.detach()

        aug_targets = \
            torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(t_cos_similarities, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.ss_ratio)
        indices = indices[:correct_num+wrong_keep]
        distill_index_ss = torch.sort(indices)[0]

        ss_loss = self.kldiv_loss(torch.log_softmax(s_cos_similarities[distill_index_ss] / self.ss_temp, dim=1),
                                  torch.softmax(t_cos_similarities[distill_index_ss] / self.ss_temp, dim=1))
        ss_loss *= (self.ss_temp ** 2)
        log_aug_outputs = torch.log_softmax(student_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        tf_loss = self.kldiv_loss(log_aug_outputs[distill_index_tf], aug_knowledges[distill_index_tf])
        tf_loss *= (self.tf_temp ** 2)
        total_loss = 0
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, ss_loss, tf_loss]):
            total_loss += loss_weight * loss
        return total_loss
