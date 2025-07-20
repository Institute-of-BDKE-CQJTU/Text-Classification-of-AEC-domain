# focalloss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):

                self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, target):

        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)  # = softmax(logits, dim=1)

        log_prob_gt = log_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        prob_gt = prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:

            alpha_t = self.alpha[target]
            loss = - alpha_t * (1 - prob_gt) ** self.gamma * log_prob_gt
        else:
            loss = - (1 - prob_gt) ** self.gamma * log_prob_gt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
