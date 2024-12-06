import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()
        expanded_filters = n_filters * 2  # Channel expansion
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_filters, expanded_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(expanded_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(expanded_filters, expanded_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(expanded_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(expanded_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters)
        )
        
        # Scaling factor for residual
        self.res_scale = 0.2
        
    def forward(self, x):
        return x + self.res_scale * self.conv_block(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced attention mechanism
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class ProposedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=23, scale_factor=4):
        super(ProposedGenerator, self).__init__()
        
        # Enhanced initial feature extraction
        self.initial_feature = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks with channel attention and dense connections
        self.residual_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            block = nn.Sequential(
                ResidualBlock(n_filters),
                ChannelAttention(n_filters),
                nn.Dropout2d(0.1)
            )
            self.residual_blocks.append(block)
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(n_filters * n_blocks, n_filters, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        )
        
        # Global skip connection conv
        self.global_skip_conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        
        # Enhanced progressive upsampling
        self.upsampling = nn.ModuleList([])
        log_scale = int(torch.log2(torch.tensor(scale_factor)))
        for _ in range(log_scale):
            self.upsampling.extend([
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                ChannelAttention(n_filters)
            ])
        
        # Enhanced final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters // 2, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial_feature(x)
        
        # Dense feature extraction
        features = initial_features
        block_outputs = []
        
        for block in self.residual_blocks:
            features = block(features)
            block_outputs.append(features)
            
        # Multi-scale fusion
        dense_features = torch.cat(block_outputs, dim=1)
        fused_features = self.fusion(dense_features)
        
        # Global residual learning
        global_features = self.global_skip_conv(fused_features)
        up_features = global_features + initial_features
        
        # Progressive upsampling
        for up_block in self.upsampling:
            up_features = up_block(up_features)
            
        return self.final(up_features)

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, sr, hr):
        # Enhanced pixel loss
        pixel_loss = 0.9 * self.l1_loss(sr, hr) + 0.1 * self.mse_loss(sr, hr)
        
        # reduced Multi-level perceptual loss
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        perceptual_loss = (
            0.5 * self.mse_loss(sr_features, hr_features) +
            0.5 * self.l1_loss(sr_features, hr_features)
        )
        
        return pixel_loss + 0.01 * perceptual_loss

class ProposedDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(ProposedDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        # Enhanced feature extraction
        self.feature_extraction = nn.Sequential(
            discriminator_block(input_channels, 64, stride=2, normalize=False),
            discriminator_block(64, 128, stride=2),
            discriminator_block(128, 256, stride=2),
            discriminator_block(256, 512, stride=1),
            discriminator_block(512, 512, stride=1),
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        features = self.feature_extraction(x)
        validity = self.classifier(features)
        return validity.view(-1)
