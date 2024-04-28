import torch
import torch.nn as nn
import torch.nn.functional as F


class HSAKD(nn.Module):
    def __init__(self,T):
        super(HSAKD, self).__init__()
        self.kl=nn.KLDivLoss(reduction="batchmean")
        self.ce=nn.CrossEntropyLoss(reduction="mean")
        self.temperature=T
    def forward(self,ss_logits,t_ss_logits,logits,t_logits,target):

        loss_cls = torch.tensor(0.).cuda()
        loss_div = torch.tensor(0.).cuda()
        loss_cls = loss_cls + self.ce(logits[0::4], target)
        for i in range(len(ss_logits)):
            loss_div = loss_div + self.kl(torch.log_softmax(ss_logits[i]/self.temperature,1),torch.softmax( t_ss_logits[i].detach()/self.temperature,1))* (self.temperature ** 2)
        loss_div = loss_div +  self.kl(torch.log_softmax(logits/self.temperature,1),torch.softmax(t_logits.detach()/self.temperature,1))* (self.temperature ** 2)
        loss = loss_cls + loss_div
        return loss