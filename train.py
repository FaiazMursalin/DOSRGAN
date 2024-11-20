import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models import ImprovedSRCNN, Discriminator, ContentLoss


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_img = Image.open(self.lr_dir / img_name).convert('RGB')
        hr_img = Image.open(self.hr_dir / img_name).convert('RGB')
        transform = transforms.ToTensor()
        return transform(lr_img), transform(hr_img)


def gaussian_kernel(size=11, sigma=1.5):
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    gauss = np.exp(-0.5 * np.square(x / sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)


def calculate_ssim(img1, img2, window_size=11):
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    kernel = gaussian_kernel(window_size)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(img1.device)
    kernel = kernel.expand(3, 1, window_size, window_size)

    mu1 = nn.functional.conv2d(img1, kernel, padding=window_size // 2, groups=3)
    mu2 = nn.functional.conv2d(img2, kernel, padding=window_size // 2, groups=3)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=3) - mu12

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean()


def calculate_ifc(img1, img2, window_size=11):
    """
    Calculate Information Fidelity Criterion (IFC) between two images
    """
    # Convert to YCbCr and use Y channel only
    img1_y = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
    img2_y = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]

    # Create sliding windows
    kernel = torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2)

    # Calculate local means
    mu1 = nn.functional.conv2d(img1_y.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)
    mu2 = nn.functional.conv2d(img2_y.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)

    # Calculate local variances
    sigma1_sq = nn.functional.conv2d(img1_y.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu1 ** 2
    sigma2_sq = nn.functional.conv2d(img2_y.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu2 ** 2
    sigma12 = nn.functional.conv2d((img1_y * img2_y).unsqueeze(0).unsqueeze(0), kernel,
                                   padding=window_size // 2) - mu1 * mu2

    # Constants to avoid division by zero
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Calculate mutual information
    numerator = torch.log2(1 + (sigma12 ** 2 + C2) / (sigma1_sq * sigma2_sq + C1))

    # Return mean IFC value
    return numerator.mean()


def train_gan_model(generator, discriminator, train_loader, val_loader, num_epochs, device, scale_factor):
    # Loss functions
    content_criterion = ContentLoss().to(device)
    adversarial_criterion = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    # Schedulers
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-7)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-7)

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_ifc': []
    }

    best_psnr = 0
    model_save_path = f'improved_srcnn_gan_x{scale_factor}.pth'

    print("Training started...")
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

            ######################
            # Train Discriminator
            ######################
            optimizer_D.zero_grad()

            # Real images
            real_output = discriminator(hr_imgs)
            d_loss_real = adversarial_criterion(real_output, real_label)

            # Fake images
            with torch.no_grad():
                sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs.detach())
            d_loss_fake = adversarial_criterion(fake_output, fake_label)

            # Combined D loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            ######################
            # Train Generator
            ######################
            optimizer_G.zero_grad()

            sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs)

            # Calculate losses
            content_loss = content_criterion(sr_imgs, hr_imgs)
            adversarial_loss = adversarial_criterion(fake_output, real_label)
            g_loss = content_loss + 0.001 * adversarial_loss

            g_loss.backward()
            optimizer_G.step()

            # Update progress bar
            total_train_loss_G += g_loss.item()
            total_train_loss_D += d_loss.item()
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })

        scheduler_G.step()
        scheduler_D.step()

        # Calculate average losses
        avg_train_loss_G = total_train_loss_G / len(train_loader)
        avg_train_loss_D = total_train_loss_D / len(train_loader)
        history['train_loss_G'].append(avg_train_loss_G)
        history['train_loss_D'].append(avg_train_loss_D)

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
                    ssim_val = calculate_ssim(sr_imgs[i], hr_imgs[i])
                    ifc_val = calculate_ifc(sr_imgs[i], hr_imgs[i])
                    total_val_psnr += psnr_val
                    total_val_ssim += ssim_val
                    total_val_ifc += ifc_val

        # Calculate averages
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / len(val_loader.dataset)
        avg_val_ssim = total_val_ssim / len(val_loader.dataset)
        avg_val_ifc = total_val_ifc / len(val_loader.dataset)

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

    plt.figure(figsize=(25, 5))

    # Generator and Discriminator Loss
    plt.subplot(1, 5, 1)
    plt.plot(epochs, history['train_loss_G'], 'b-', label='Generator Loss')
    plt.plot(epochs, history['train_loss_D'], 'r-', label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation Loss
    plt.subplot(1, 5, 2)
    plt.plot(epochs, history['val_loss'], 'g-')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # PSNR
    plt.subplot(1, 5, 3)
    plt.plot(epochs, history['val_psnr'], 'm-')
    plt.title('PSNR on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')

    # SSIM
    plt.subplot(1, 5, 4)
    plt.plot(epochs, history['val_ssim'], 'y-')
    plt.title('SSIM on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    # IFC
    plt.subplot(1, 5, 5)
    plt.plot(epochs, history['val_ifc'], 'c-')
    plt.title('IFC on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('IFC')

    plt.tight_layout()
    plt.savefig('training_curves_gan.png')
    plt.close()


def save_metrics(history, filename='metrics_history_gan.txt'):
    """Save metrics to a file"""
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
    # Configuration
    scale_factor = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 300

    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"- Scale factor: {scale_factor}x")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {num_epochs}")

    # Create generator and discriminator
    generator = ImprovedSRCNN(scale_factor=scale_factor).to(device)
    discriminator = Discriminator().to(device)

    # Dataset paths
    train_hr_dir = './dataset_improved/general100/x2/hr'
    train_lr_dir = './dataset_improved/general100/x2/lr'
    val_hr_dir = './dataset_improved/set5/x2/hr'
    val_lr_dir = './dataset_improved/set5/x2/lr'

    # Check if directories exist
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Create datasets and dataloaders
    train_dataset = SRDataset(train_hr_dir, train_lr_dir)
    val_dataset = SRDataset(val_hr_dir, val_lr_dir)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    # Train the model
    history = train_gan_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        scale_factor=scale_factor
    )

    # Plot and save results
    plot_training_curves(history)
    save_metrics(history)

    print("Training completed!")
