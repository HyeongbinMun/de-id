import torch
import torch.nn as nn
from torchvision import models


class ResNet50Inverter(nn.Module):
    def __init__(self):
        super(ResNet50Inverter, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            *list(resnet50.children())[:-2])  # Removing the last FC layer and global average pooling

        # Define the inversion part of the model
        self.inversion_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using ResNet50
        features = self.features(x)
        # Invert features to reconstruct the input
        reconstructed = self.inversion_layers(features)
        return reconstructed