import torch,math
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    """
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py
    """

    def init_prob_alias(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())

        k = len(probs)
        self.probs = torch.zeros(k)
        self.alias = torch.zeros(k, dtype=torch.int64)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.probs[kk] = k * prob
            if self.probs[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.probs[large] = (self.probs[large] - 1.0) + self.probs[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.probs[last_one] = 1

    def __init__(self, input_size, output_size, num_negative_samples, num_samples, temperature=0.07, momentum=0.5, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.unigrams = torch.ones(output_size)
        self.num_negative_samples = num_negative_samples
        self.num_samples = num_samples
        self.register_buffer('params', torch.tensor([num_negative_samples, temperature, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(input_size / 3)
        self.register_buffer('memory_v1', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        self.cross_entropy_loss=nn.CrossEntropyLoss(reduction="mean")
        self.probs, self.alias = None, None
        self.init_prob_alias(self.unigrams)

    def draw(self, n):
        """ Draw n samples from multinomial """
        k = self.alias.size(0)
        kk = torch.zeros(n, dtype=torch.long, device=self.prob.device).random_(0, k)
        prob = self.probs.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())
        return oq + oj

    def contrast_memory(self, student_embed, teacher_embed, pos_indices, contrast_idx=None):
        param_k = int(self.params[0].item())
        param_t = self.params[1].item()
        z_v1 = self.params[2].item()
        z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batch_size = student_embed.size(0)
        output_size = self.memory_v1.size(0)
        input_size = self.memory_v1.size(1)

        # original score computation
        if contrast_idx is None:
            contrast_idx = self.draw(batch_size * (self.num_negative_samples + 1)).view(batch_size, -1)
            contrast_idx.select(1, 0).copy_(pos_indices.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, contrast_idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batch_size, param_k + 1, input_size)
        out_v2 = torch.bmm(weight_v1, teacher_embed.view(batch_size, input_size, 1))
        out_v2 = torch.exp(torch.div(out_v2, param_t))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, contrast_idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batch_size, param_k + 1, input_size)
        out_v1 = torch.bmm(weight_v2, student_embed.view(batch_size, input_size, 1))
        out_v1 = torch.exp(torch.div(out_v1, param_t))

        # set z if haven't been set yet
        if z_v1 < 0:
            self.params[2] = out_v1.mean() * output_size
            z_v1 = self.params[2].clone().detach().item()
        if z_v2 < 0:
            self.params[3] = out_v2.mean() * output_size
            z_v2 = self.params[3].clone().detach().item()

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, z_v1).contiguous()
        out_v2 = torch.div(out_v2, z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, pos_indices.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(student_embed, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, pos_indices, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, pos_indices.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(teacher_embed, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, pos_indices, updated_v2)
        return out_v1, out_v2

    def compute_contrast_loss(self, x):
        batch_size = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        pn = 1 / float(self.num_samples)

        # loss for positive pair
        p_pos = x.select(1, 0)
        log_d1 = torch.div(p_pos, p_pos.add(m * pn + self.eps)).log_()

        # loss for K negative pair
        p_neg = x.narrow(1, 1, m)
        log_d0 = torch.div(p_neg.clone().fill_(m * pn), p_neg.add(m * pn + self.eps)).log_()

        loss = - (log_d1.sum(0) + log_d0.view(-1, 1).sum(0)) / batch_size
        return loss

    def forward(self, nornamize_s_out,normalize_t_out,pos_idx,contrast_idx,linear_s_out,linear_t_out,targets):
        """
        pos_idx: the indices of these positive samples in the dataset, size [batch_size]
        contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
        device = nornamize_s_out.device
        pos_idx = pos_idx.to(device)
        if contrast_idx is not None:
            contrast_idx = contrast_idx.to(device)
        if device != self.probs.device:
            self.probs.to(device)
            self.alias.to(device)
            self.to(device)
        out_s, out_t = self.contrast_memory(nornamize_s_out, normalize_t_out, pos_idx, contrast_idx)
        student_contrast_loss = self.compute_contrast_loss(out_s)
        teacher_contrast_loss = self.compute_contrast_loss(out_t)
        crd_loss = student_contrast_loss + teacher_contrast_loss
        soft_loss = self.kl_loss(torch.log_softmax(linear_s_out / 4, dim=1),
                                    torch.softmax(linear_t_out / 4, dim=1))
        hard_loss = self.cross_entropy_loss(linear_s_out, targets)
        vanilla_kd_loss=hard_loss + (4** 2) * soft_loss
        return vanilla_kd_loss*1.0+crd_loss*0.8