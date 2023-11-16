import random
import torch
import numpy as np
from torchvision.transforms import functional as F

def tensor_to_image(tensor):
    return F.to_pil_image(tensor)

def seed_random(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)