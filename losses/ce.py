import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)
