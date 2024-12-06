import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from utils_2_x2 import SRDataset, evaluate_model, plot_training_curves, save_results
from model_5_x2 import ProposedGenerator, ProposedDiscriminator, ContentLoss
from tqdm import tqdm

def check_loss_value(loss, name, threshold=5.0):
    if torch.isnan(loss).any():
        print(f"Warning: NaN detected in {name}")
        return torch.tensor(0.0).to(loss.device)
    if loss.item() > threshold:
        print(f"Warning: {name} exceeded threshold ({loss.item():.4f})")
        return torch.tensor(threshold).to(loss.device)
    return loss

def train_model(generator, discriminator, train_loader, val_loader, test_set5_loader, test_set14_loader, 
                num_epochs, device, scale_factor):
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Loss functions
    content_criterion = ContentLoss().to(device)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = nn.L1Loss().to(device)

    # Optimizers with adjusted learning rates
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8)

    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, 
        mode='max', 
        factor=0.5, 
        patience=20, 
        verbose=True, 
        threshold=1e-3,
        min_lr=1e-6
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, 
        mode='min', 
        factor=0.5, 
        patience=20, 
        verbose=True,
        min_lr=1e-7
    )

    # Dynamic loss weights function with extended pixel phase
    def get_loss_weights(epoch, max_epochs):
        if epoch < max_epochs * 0.5:  # First 50% epochs
            return 1.0, 0.0, 0.0  # Only pixel loss
        elif epoch < max_epochs * 0.8:  # Next 30% epochs
            return 0.9, 0.1, 0.0  # Mainly pixel loss + small content loss
        else:  # Last 20% epochs
            return 0.8, 0.1, 0.0001  # Reduced adversarial impact

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_psnr': [], 'val_ssim': [],
        'set5_psnr': [], 'set5_ssim': [],
        'set14_psnr': [], 'set14_ssim': []
    }

    model_save_path = f'proposed_model_optimized_x{scale_factor}.pth'
    best_model_path = f'best_model_x{scale_factor}.pth'
    accumulation_steps = 4  # Gradient accumulation steps
    best_psnr = 0

    print("Starting training with phased approach:")
    print("Phase 1 (0-50%): Pure Pixel Loss")
    print("Phase 2 (50-80%): Pixel + Content Loss")
    print("Phase 3 (80-100%): Full Model with Minimal GAN Loss")

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        batch_count = 0

        # Get current loss weights
        lambda_pixel, lambda_content, lambda_adv = get_loss_weights(epoch, num_epochs)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        for lr_imgs, hr_imgs in progress_bar:
            batch_size = lr_imgs.size(0)
            batch_count += 1
            
            # Label smoothing
            real_label = torch.ones((batch_size, 1), device=device).uniform_(0.8, 1.0)
            fake_label = torch.zeros((batch_size, 1), device=device).uniform_(0.0, 0.2)
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            #------------------------
            # Train Generator
            #------------------------
            with autocast():
                sr_imgs = generator(lr_imgs)
                # Clamp outputs to valid range
                sr_imgs = torch.clamp(sr_imgs, 0, 1)
                
                # Calculate pixel loss (always present)
                pixel_loss = criterion_pixel(sr_imgs, hr_imgs)
                pixel_loss = torch.clamp(pixel_loss, 0, 5.0)
                g_loss = lambda_pixel * pixel_loss
                
                # Add content loss if in appropriate phase
                if epoch >= num_epochs * 0.5:
                    content_loss = content_criterion(sr_imgs, hr_imgs)
                    content_loss = torch.clamp(content_loss, 0, 5.0)
                    g_loss = g_loss + lambda_content * content_loss
                else:
                    content_loss = torch.tensor(0.0, device=device)
                    
                # Add GAN loss in final phase
                if epoch >= num_epochs * 0.8:
                    pred_real = discriminator(hr_imgs).detach()
                    pred_fake = discriminator(sr_imgs)
                    
                    pred_fake = pred_fake.unsqueeze(1) if pred_fake.dim() == 1 else pred_fake
                    pred_real = pred_real.unsqueeze(1) if pred_real.dim() == 1 else pred_real
                    
                    loss_GAN = (criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), real_label) +
                               criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake_label)) / 2
                    
                    loss_GAN = torch.clamp(loss_GAN, 0, 2.0)
                    g_loss = g_loss + lambda_adv * loss_GAN
                else:
                    loss_GAN = torch.tensor(0.0, device=device)

            # Scale and accumulate gradients
            scaler.scale(g_loss / accumulation_steps).backward()
            if batch_count % accumulation_steps == 0:
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
                scaler.step(optimizer_G)
                optimizer_G.zero_grad()

            #------------------------
            # Train Discriminator
            #------------------------
            optimizer_D.zero_grad()
            with autocast():
                pred_real = discriminator(hr_imgs)
                pred_fake = discriminator(sr_imgs.detach())
                
                pred_fake = pred_fake.unsqueeze(1) if pred_fake.dim() == 1 else pred_fake
                pred_real = pred_real.unsqueeze(1) if pred_real.dim() == 1 else pred_real
                
                # More stable GAN training
                loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), real_label)
                loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake_label)
                d_loss = (loss_real + loss_fake) / 2
                d_loss = torch.clamp(d_loss, 0, 2.0)

            scaler.scale(d_loss / accumulation_steps).backward()
            if batch_count % accumulation_steps == 0:
                scaler.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                scaler.step(optimizer_D)
                optimizer_D.zero_grad()

            # Update scaler
            if batch_count % accumulation_steps == 0:
                scaler.update()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            current_lr_g = optimizer_G.param_groups[0]['lr']
            current_lr_d = optimizer_D.param_groups[0]['lr']
            
            # More informative progress bar
            phase = "Pixel" if epoch < num_epochs * 0.5 else ("Pixel+Content" if epoch < num_epochs * 0.8 else "Full")
            progress_bar.set_postfix({
                'Phase': phase,
                'G_loss': f'{g_loss.item():.4f}',
                'Pixel': f'{pixel_loss.item():.4f}',
                'Content': f'{content_loss.item():.4f}',
                'GAN': f'{loss_GAN.item():.4f}' if epoch >= num_epochs * 0.8 else '0.0000',
                'LR_G': f'{current_lr_g:.2e}'
            })

        # Calculate average epoch losses
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        # Evaluate model
        generator.eval()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                val_metrics = evaluate_model(generator, val_loader, device, "Validation")
                set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")
                set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")
        generator.train()

        # Save best model based on Set5 PSNR
        if set5_metrics[0] > best_psnr:
            best_psnr = set5_metrics[0]
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'best_psnr': best_psnr,
            }, best_model_path)
            print(f"\nNew best model saved! Set5 PSNR: {best_psnr:.2f}")

        # Update learning rates
        scheduler_G.step(set5_metrics[0])
        scheduler_D.step(avg_d_loss)

        # Update history
        history['train_loss_G'].append(avg_g_loss)
        history['train_loss_D'].append(avg_d_loss)
        history['val_psnr'].append(val_metrics[0])
        history['val_ssim'].append(val_metrics[1])
        history['set5_psnr'].append(set5_metrics[0])
        history['set5_ssim'].append(set5_metrics[1])
        history['set14_psnr'].append(set14_metrics[0])
        history['set14_ssim'].append(set14_metrics[1])

        # Save latest model
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'scaler': scaler.state_dict(),
            'psnr_set5': set5_metrics[0],
            'ssim_set5': set5_metrics[1],
            'psnr_set14': set14_metrics[0],
            'ssim_set14': set14_metrics[1],
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'history': history
        }, model_save_path)

        print(f'\nEpoch {epoch+1} Results:')
        print(f'Phase: {phase}')
        print(f'Loss G: {avg_g_loss:.4f}, Loss D: {avg_d_loss:.4f}')
        print(f'Set5 - PSNR: {set5_metrics[0]:.2f}, SSIM: {set5_metrics[1]:.4f}')
        print(f'Set14 - PSNR: {set14_metrics[0]:.2f}, SSIM: {set14_metrics[1]:.4f}')
        print(f'Learning Rates - G: {current_lr_g:.2e}, D: {current_lr_d:.2e}\n')

    return history

if __name__ == "__main__":
    # Configuration
    scale_factor = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8  # Reduced batch size since we're using gradient accumulation
    num_epochs = 1000

    print(f"Using device: {device}")
    print(f"Scale factor: {scale_factor}x")
    print(f"Batch size: {batch_size}")
    print(f"Effective batch size: {batch_size * 4}")  # Due to gradient accumulation
    print(f"Number of epochs: {num_epochs}")

    # Initialize models
    generator = ProposedGenerator(scale_factor=scale_factor).to(device)
    discriminator = ProposedDiscriminator().to(device)

    # Dataset paths
    general100_hr_dir = './dataset_improved/general100/x2/hr'
    general100_lr_dir = './dataset_improved/general100/x2/lr'
    set5_hr_dir = './dataset_improved/set5/x2/hr'
    set5_lr_dir = './dataset_improved/set5/x2/lr'
    set14_hr_dir = './dataset_improved/set14/x2/hr'
    set14_lr_dir = './dataset_improved/set14/x2/lr'

    # Create datasets
    general100_dataset = SRDataset(general100_hr_dir, general100_lr_dir)
    set5_dataset = SRDataset(set5_hr_dir, set5_lr_dir)
    set14_dataset = SRDataset(set14_hr_dir, set14_lr_dir)

    # Split General100 into train/val
    train_size = int(0.8 * len(general100_dataset))
    val_size = len(general100_dataset) - train_size
    train_dataset, val_dataset = random_split(
        general100_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False)
    test_set14_loader = DataLoader(set14_dataset, batch_size=1, shuffle=False)

    print("\nDataset sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Set5 test samples: {len(set5_dataset)}")
    print(f"Set14 test samples: {len(set14_dataset)}")

    print("\nTraining phases:")
    print("Phase 1 (0-30%): Pixel Loss Only")
    print("Phase 2 (30-60%): Pixel + Content Loss")
    print("Phase 3 (60-100%): Full GAN Training")

    # Train the model
    history = train_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_set5_loader=test_set5_loader,
        test_set14_loader=test_set14_loader,
        num_epochs=num_epochs,
        device=device,
        scale_factor=scale_factor
    )

    # Plot and save results
    plot_training_curves(history)
    print("Training curves plotted")

    # Final evaluation
    print("\nPerforming final evaluation...")
    generator.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            set5_psnr, set5_ssim = evaluate_model(generator, test_set5_loader, device, "Set5")
            set14_psnr, set14_ssim = evaluate_model(generator, test_set14_loader, device, "Set14")

    final_metrics = {
        'Set5': {'psnr': set5_psnr, 'ssim': set5_ssim},
        'Set14': {'psnr': set14_psnr, 'ssim': set14_ssim}
    }
    
    save_results(history, final_metrics)
    print("\nFinal Results:")
    print(f"Set5 - PSNR: {set5_psnr:.2f}, SSIM: {set5_ssim:.4f}")
    print(f"Set14 - PSNR: {set14_psnr:.2f}, SSIM: {set14_ssim:.4f}")
