
import torch

from model.vcd.vcd.models.frame import L2N, BaseModel

SEGMENT_MODELS = ['Segment_MaxPool', 'Segment_AvgPool']

class Segment_MaxPool(BaseModel):
    def __init__(self):
        super(Segment_MaxPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x, _ = torch.max(x, 1)
        x = self.norm(x)
        return x


class Segment_AvgPool(BaseModel):
    def __init__(self):
        super(Segment_AvgPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x, _ = torch.mean(x, 1)
        x = self.norm(x)
        return x