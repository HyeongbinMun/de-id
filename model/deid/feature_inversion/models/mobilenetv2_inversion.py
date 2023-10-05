import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2Inverter(nn.Module):
    def __init__(self):
        super(MobileNetV2Inverter, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True).features
        # Define the inversion part of the model
        self.inversion_layers = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using MobileNetV2
        features = self.mobilenet_v2(x)
        # Invert features to reconstruct the input
        reconstructed = self.inversion_layers(features)
        return reconstructed