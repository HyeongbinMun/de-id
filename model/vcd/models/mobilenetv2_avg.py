import torch
from torchvision import models
from collections import OrderedDict


class L2N(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


class MobileNet_AVG(torch.nn.Module):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = torch.nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=False).features.named_children())]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x