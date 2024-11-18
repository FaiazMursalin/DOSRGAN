import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseBlock(nn.Module):
    def __init__(self, n_filters):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_filters * 3, n_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters, kernel_size=3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.bn4 = nn.BatchNorm2d(n_filters)

        self.se = SELayer(n_filters)

    def forward(self, x):
        identity = x

        out1 = self.lrelu(self.bn1(self.conv1(x)))
        cat1 = torch.cat([x, out1], 1)

        out2 = self.lrelu(self.bn2(self.conv2(cat1)))
        cat2 = torch.cat([cat1, out2], 1)

        out3 = self.lrelu(self.bn3(self.conv3(cat2)))
        cat3 = torch.cat([cat2, out3], 1)

        out4 = self.lrelu(self.bn4(self.conv4(cat3)))
        out4 = self.se(out4)

        return out4 + identity + out1 + out2 + out3


class DOSRGAN(nn.Module):
    def __init__(self, scale_factor=2, n_filters=128, n_blocks=16):
        super(DOSRGAN, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.residual_dense_blocks = nn.ModuleList([
            DenseBlock(n_filters) for _ in range(n_blocks)
        ])

        self.skip_conv1 = nn.Conv2d(n_filters * n_blocks, n_filters, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=1)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 1),
            nn.LeakyReLU(0.2, inplace=True),
            SELayer(n_filters)
        )

        self.global_fusion = nn.Sequential(
            nn.Conv2d(n_filters * n_blocks, n_filters, kernel_size=1),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),
            SELayer(n_filters)
        )

        n_upsamples = int(np.log2(scale_factor))
        self.upsampling = nn.ModuleList()
        for _ in range(n_upsamples):
            self.upsampling.append(nn.Sequential(
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                SELayer(n_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        self.final = nn.Sequential(
            nn.Conv2d(n_filters, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        initial = self.initial_conv(x)
        x = initial

        block_features = []
        for block in self.residual_dense_blocks:
            x = block(x)
            block_features.append(x)

        cat_features = torch.cat(block_features, dim=1)
        global_features = self.global_fusion(cat_features)

        skip_path = self.skip_conv1(cat_features)
        skip_path = torch.cat([skip_path, initial], dim=1)
        skip_path = self.skip_conv2(skip_path)

        x = global_features + skip_path
        x = self.channel_attention(x)

        for up_block in self.upsampling:
            x = up_block(x)

        return (self.final(x) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, stride=2, batch_norm=False),
            *discriminator_block(64, 128, stride=2),
            *discriminator_block(128, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:36]
        self.vgg = nn.Sequential(*[vgg[i] for i in range(36)]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        vgg_sr = self.vgg(sr)
        vgg_hr = self.vgg(hr)
        return F.mse_loss(vgg_sr, vgg_hr)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()

    def forward(self, sr, hr):
        l1_loss = self.l1_loss(sr, hr)
        vgg_loss = self.vgg_loss(sr, hr)
        return 0.5 * l1_loss + 0.5 * vgg_loss