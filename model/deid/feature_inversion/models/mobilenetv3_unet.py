import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large

class ResidualUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        skip_connection = self.conv1x1(skip_connection)
        x = torch.cat([x, skip_connection], dim=1)
        return x

class MobileNetV3SmallUNetInverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(pretrained=True).features

        self.up1 = ResidualUpsample(576, 256)
        self.up2 = ResidualUpsample(256, 128)
        self.up3 = ResidualUpsample(128, 64)
        self.up4 = ResidualUpsample(64, 32)
        self.up5 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.mobilenet):
            x = layer(x)
            if i in [1, 4, 7, 9]:
                features.append(x)
        features.reverse()

        out = self.up1(features[0], features[1])
        out = self.up2(out, features[2])
        out = self.up3(out, features[3])
        out = self.up4(out, x)
        out = self.up5(out)

        return out

class MobileNetV3LargeUNetInverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v3_large(pretrained=True).features

        self.up1 = ResidualUpsample(960, 512)
        self.up2 = ResidualUpsample(512, 256)
        self.up3 = ResidualUpsample(256, 128)
        self.up4 = ResidualUpsample(128, 64)
        self.up5 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.mobilenet):
            x = layer(x)
            if i in [4, 7, 10, 14]:
                features.append(x)
        features.reverse()

        out = self.up1(features[0], features[1])
        out = self.up2(out, features[2])
        out = self.up3(out, features[3])
        out = self.up4(out, x)
        out = self.up5(out)

        return out