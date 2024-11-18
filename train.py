import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from math import exp
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import DOSRGAN, Discriminator
from torch.utils.data import DataLoader
import os



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
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.se = SELayer(n_filters)

    def forward(self, x):
        out1 = self.lrelu(self.bn1(self.conv1(x)))
        cat1 = torch.cat([x, out1], 1)
        out2 = self.lrelu(self.bn2(self.conv2(cat1)))
        cat2 = torch.cat([cat1, out2], 1)
        out3 = self.lrelu(self.bn3(self.conv3(cat2)))
        out3 = self.se(out3)
        return out3 + x


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
        x = self.initial_conv(x)
        residual = x

        features = []
        for block in self.residual_dense_blocks:
            x = block(x)
            features.append(x)

        x = torch.cat(features, dim=1)
        x = self.global_fusion(x)
        x = self.channel_attention(x)
        x = x + residual

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
            nn.Sigmoid()
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
        return 0.5 * self.l1_loss(sr, hr) + 0.5 * self.vgg_loss(sr, hr)





class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, is_train=True):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_img = Image.open(self.lr_dir / img_name).convert('RGB')
        hr_img = Image.open(self.hr_dir / img_name).convert('RGB')

        if self.is_train:
            seed = torch.randint(0, 2 ** 32, (1,))[0].item()
            torch.manual_seed(seed)
            lr_img = self.transform(lr_img)
            torch.manual_seed(seed)
            hr_img = self.transform(hr_img)
        else:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0) - 10 * torch.log10(mse)


def calculate_ssim(img1, img2, window_size=11):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ifc(img1, img2):
    # Simplified IFC calculation
    return F.mse_loss(img1, img2).item() * -1



def train_gan_model(generator, discriminator, train_loader, val_loader, num_epochs, device, scale_factor):
    content_criterion = ContentLoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    scaler = torch.amp.GradScaler('cuda')

    def warmup_lr_scheduler(epoch):
        if epoch < 10:
            return epoch / 10
        return 0.5 * (1 + np.cos((epoch - 10) * np.pi / (num_epochs - 10)))

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, warmup_lr_scheduler)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, warmup_lr_scheduler)

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_ifc': []
    }

    best_psnr = 0
    model_save_path = f'improved_srcnn_gan_x{scale_factor}.pth'

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_train_loss_G = 0
        total_train_loss_D = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for lr_imgs, hr_imgs in progress_bar:
            batch_size = lr_imgs.size(0)
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train Discriminator
            with torch.cuda.amp.autocast('cuda'):
                optimizer_D.zero_grad()
                real_output = discriminator(hr_imgs)
                d_loss_real = adversarial_criterion(real_output, real_label)

                sr_imgs = generator(lr_imgs)
                fake_output = discriminator(sr_imgs.detach())
                d_loss_fake = adversarial_criterion(fake_output, fake_label)

                d_loss = (d_loss_real + d_loss_fake) * 0.5

            scaler.scale(d_loss).backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer_D)

            # Train Generator
            with torch.cuda.amp.autocast('cuda'):
                optimizer_G.zero_grad()
                sr_imgs = generator(lr_imgs)
                fake_output = discriminator(sr_imgs)

                content_loss = content_criterion(sr_imgs, hr_imgs)
                adversarial_loss = adversarial_criterion(fake_output, real_label)
                g_loss = content_loss + 0.001 * adversarial_loss

            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(optimizer_G)
            scaler.update()

            total_train_loss_G += g_loss.item()
            total_train_loss_D += d_loss.item()
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })

        scheduler_G.step()
        scheduler_D.step()

        # Validation phase
        generator.eval()
        total_val_loss = 0
        total_val_psnr = 0
        total_val_ssim = 0
        total_val_ifc = 0

        print("\nValidating...")
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(val_loader, desc="Validation"):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)

                val_loss = content_criterion(sr_imgs, hr_imgs)
                total_val_loss += val_loss.item()

                for i in range(sr_imgs.size(0)):
                    psnr_val = calculate_psnr(sr_imgs[i], hr_imgs[i])
                    ssim_val = calculate_ssim(sr_imgs[i].unsqueeze(0), hr_imgs[i].unsqueeze(0))
                    ifc_val = calculate_ifc(sr_imgs[i].unsqueeze(0), hr_imgs[i].unsqueeze(0))
                    total_val_psnr += psnr_val
                    total_val_ssim += ssim_val
                    total_val_ifc += ifc_val

        # Calculate averages
        avg_train_loss_G = total_train_loss_G / len(train_loader)
        avg_train_loss_D = total_train_loss_D / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / len(val_loader.dataset)
        avg_val_ssim = total_val_ssim / len(val_loader.dataset)
        avg_val_ifc = total_val_ifc / len(val_loader.dataset)

        # Update history
        history['train_loss_G'].append(avg_train_loss_G)
        history['train_loss_D'].append(avg_train_loss_D)
        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_val_psnr)
        history['val_ssim'].append(avg_val_ssim)
        history['val_ifc'].append(avg_val_ifc)

        print(f'\nEpoch {epoch + 1}/{num_epochs} Results:')
        print(f'G Loss: {avg_train_loss_G:.6f}')
        print(f'D Loss: {avg_train_loss_D:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'Val PSNR: {avg_val_psnr:.2f}')
        print(f'Val SSIM: {avg_val_ssim:.4f}')
        print(f'Val IFC: {avg_val_ifc:.4f}')

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': avg_val_ssim,
                'best_ifc': avg_val_ifc,
            }, model_save_path)
            print(f'Saved model with PSNR: {best_psnr:.2f}, SSIM: {avg_val_ssim:.4f}, IFC: {avg_val_ifc:.4f}')

    return history


def plot_training_curves(history):
    epochs = range(1, len(history['train_loss_G']) + 1)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss_G'], 'b-', label='Generator Loss')
    plt.plot(epochs, history['train_loss_D'], 'r-', label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['val_loss'], 'g-')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['val_psnr'], 'm-')
    plt.title('PSNR on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')

    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['val_ssim'], 'y-')
    plt.title('SSIM on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['val_ifc'], 'c-')
    plt.title('IFC on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('IFC')

    plt.tight_layout()
    plt.savefig('training_curves_gan.png')
    plt.close()


def save_metrics(history, filename='metrics_history_gan.txt'):
    with open(filename, 'w') as f:
        f.write("Epoch,Generator Loss,Discriminator Loss,Val Loss,PSNR,SSIM,IFC\n")
        for i in range(len(history['train_loss_G'])):
            f.write(f"{i + 1},{history['train_loss_G'][i]:.6f},"
                    f"{history['train_loss_D'][i]:.6f},"
                    f"{history['val_loss'][i]:.6f},"
                    f"{history['val_psnr'][i]:.2f},"
                    f"{history['val_ssim'][i]:.4f},"
                    f"{history['val_ifc'][i]:.4f}\n")



if __name__ == "__main__":
    scale_factor = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 3

    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"- Scale factor: {scale_factor}x")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {num_epochs}")

    # Initialize models
    generator = DOSRGAN(scale_factor=scale_factor).to(device)
    discriminator = Discriminator().to(device)

    # Dataset paths
    train_hr_dir = './dataset_improved/general100/x2/hr'
    train_lr_dir = './dataset_improved/general100/x2/lr'
    val_hr_dir = './dataset_improved/set5/x2/hr'
    val_lr_dir = './dataset_improved/set5/x2/lr'

    # Check directories
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Create datasets
    train_dataset = SRDataset(train_hr_dir, train_lr_dir, is_train=True)
    val_dataset = SRDataset(val_hr_dir, val_lr_dir, is_train=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # Train model
    history = train_gan_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        scale_factor=scale_factor
    )

    # Save results
    plot_training_curves(history)
    save_metrics(history)

    print("Training completed!")