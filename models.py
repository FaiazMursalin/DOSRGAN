import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features[:36]
        self.vgg = nn.Sequential(*[vgg[i] for i in range(36)]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        vgg_sr = self.vgg(sr)
        vgg_hr = self.vgg(hr)
        return F.mse_loss(vgg_sr, vgg_hr)


class ResidualDenseBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        growth_channels = 32
        self.convs = nn.ModuleList()
        for i in range(5):
            in_channels = n_filters + i * growth_channels
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, growth_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True)
                )
            )
        self.conv_last = nn.Conv2d(n_filters + 5 * growth_channels, n_filters, 1)

    def forward(self, x):
        inputs = [x]
        for conv in self.convs:
            x = conv(torch.cat(inputs, 1))
            inputs.append(x)
        return self.conv_last(torch.cat(inputs, 1)) * 0.2 + inputs[0]


class RRDB(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.dense_blocks = nn.ModuleList([
            ResidualDenseBlock(n_filters) for _ in range(3)
        ])

    def forward(self, x):
        out = x
        for block in self.dense_blocks:
            out = block(out)
        return out * 0.2 + x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)


class HybridRRDBNet(nn.Module):
    def __init__(self, scale_factor=2, n_filters=256, n_blocks=23):
        super().__init__()
        self.scale_factor = scale_factor

        # Initial feature extraction
        self.conv_first = nn.Conv2d(3, n_filters, 3, 1, 1)

        # RRDB blocks
        self.RRDB_blocks = nn.ModuleList([
            RRDB(n_filters) for _ in range(n_blocks)
        ])

        # Channel attention blocks
        self.ca_blocks = nn.ModuleList([
            ChannelAttention(n_filters) for _ in range(4)
        ])

        # Trunk branch
        self.trunk_conv = nn.Conv2d(n_filters, n_filters, 3, 1, 1)

        # Upsampling
        n_upsamples = int(torch.log2(torch.tensor(scale_factor)))
        self.upsampling = nn.ModuleList()
        for _ in range(n_upsamples):
            self.upsampling.extend([
                nn.Conv2d(n_filters, n_filters * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, True)
            ])

        self.conv_hr = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.conv_last = nn.Conv2d(n_filters, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = fea

        # RRDB blocks with channel attention
        for i, block in enumerate(self.RRDB_blocks):
            trunk = block(trunk)
            if i % 6 == 5:  # Apply CA every 6 blocks
                ca_idx = i // 6
                if ca_idx < len(self.ca_blocks):
                    trunk = self.ca_blocks[ca_idx](trunk)

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        # Upsampling
        for up_block in self.upsampling:
            fea = up_block(fea)

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, stride=2, normalize=False),
            *discriminator_block(64, 128, stride=2),
            *discriminator_block(128, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, sr, hr, gan_loss=None):
        l1 = self.l1_loss(sr, hr)
        percep = self.perceptual_loss(sr, hr)

        if gan_loss is not None:
            # 0.8 L1 + 0.15 Perceptual + 0.05 GAN for balance
            return 0.8 * l1 + 0.15 * percep + 0.05 * gan_loss
        return l1 + 0.1 * percep


def create_model(scale_factor=2):
    generator = HybridRRDBNet(scale_factor=scale_factor)
    discriminator = Discriminator()
    content_loss = ContentLoss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    return generator, discriminator, content_loss, adversarial_loss