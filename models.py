import torch
import torch.nn as nn
import numpy as np


import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Adjust input size to account for concatenated avg_pool and max_pool outputs
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Process both average and max pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        # Concatenate the pooling results
        y = torch.cat([y_avg, y_max], dim=1)
        # Apply fully connected layers and reshape
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters * 2)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.se = SELayer(n_filters)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + identity
        out = self.relu(out)
        return out


class ImprovedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(ImprovedSRCNN, self).__init__()

        # Initial Feature Extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=4),  # changed kernel to 9 and padding to 4
            nn.ReLU(inplace=False)
        )

        # Deep residual blocks
        self.residual_layers = nn.ModuleList([
            EnhancedResidualBlock(128) for _ in range(10)  # from 8 to 10
        ])

        # Feature fusion with same padding
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            SELayer(128, reduction=8)
        )

        # Progressive upsampling
        n_upsamples = int(np.log2(scale_factor))
        self.upsampling = nn.ModuleList()
        for _ in range(n_upsamples):
            self.upsampling.append(nn.Sequential(
                nn.Conv2d(128, 512, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                SELayer(128),
                nn.ReLU(inplace=False)
            ))

        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        residual = x

        # Residual blocks with dense connections
        features = []
        for res_block in self.residual_layers:
            x = res_block(x)
            features.append(x)

        # Feature fusion
        x = torch.cat(features[-4:], dim=1)  # Use last 3 features
        x = nn.Conv2d(128 * 4, 128, kernel_size=1).to(x.device)(x)  # 1x1 conv to reduce channels
        x = self.fusion(x)

        # Global residual connection
        x = x + residual

        # Progressive upsampling
        for up_block in self.upsampling:
            x = up_block(x)

        # Final reconstruction
        return self.final(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )

        self.model = nn.Sequential(
            discriminator_block(3, 64, 2),
            discriminator_block(64, 128, 2),
            discriminator_block(128, 256, 2),
            discriminator_block(256, 512, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr):
        return 0.5 * self.l1_loss(sr, hr) + 0.5 * self.mse_loss(sr, hr)
