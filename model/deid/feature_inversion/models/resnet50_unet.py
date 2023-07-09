import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResidualUpsample(nn.Module):
    def __init__(self, input_channels, output_channels, skip_channels):
        super(ResidualUpsample, self).__init__()

        self.conv1x1 = nn.Conv2d(skip_channels, output_channels, kernel_size=1)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip_connection=None):
        x = self.upsample(x)
        x = self.conv(x)
        if skip_connection is not None:
            skip_connection = self.conv1x1(skip_connection)  # Reduce the channels
            x += F.interpolate(skip_connection, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x


class ResNet50UNetInverter(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained=True)

        self.up1 = ResidualUpsample(2048, 1024, 1024)
        self.up2 = ResidualUpsample(1024, 512, 512)
        self.up3 = ResidualUpsample(512, 256, 256)
        self.up4 = ResidualUpsample(256, 64, 256)  # Updated skip channels size
        self.up5 = ResidualUpsample(64, 3, 64)

    def forward(self, x):
        original_size = x.shape[2:]  # Save the original size

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        features = []
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            x = layer(x)
            features.append(x)

        out = self.up1(features[3], features[2])
        out = self.up2(out, features[1])
        out = self.up3(out, features[0])
        out = self.up4(out, features[0])  # use the same features[0] as skip-connection
        out = self.up5(out)

        return out