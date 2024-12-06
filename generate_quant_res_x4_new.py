import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from model_5 import ProposedGenerator
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

def load_trained_model(model_path, scale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ProposedGenerator(scale_factor=scale_factor).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load only the generator
    model.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Loaded model from epoch: {checkpoint['epoch']}")
    print(f"Set5 PSNR: {checkpoint['psnr_set5']:.2f}")
    print(f"Set5 SSIM: {checkpoint['ssim_set5']:.4f}")
    print(f"Set14 PSNR: {checkpoint['psnr_set14']:.2f}")
    print(f"Set14 SSIM: {checkpoint['ssim_set14']:.4f}")
    
    model.eval()
    return model, device

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return to_tensor(img).unsqueeze(0)

def perform_super_resolution(lr_img, model, device, method='proposed', scale_factor=2):
    with torch.no_grad():
        if method == 'proposed':
            lr_img = lr_img.to(device)
            sr_img = model(lr_img)
            sr_img = torch.clamp(sr_img, 0, 1)
            return sr_img.cpu()
        elif method == 'bilinear':
            return F.interpolate(lr_img, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        elif method == 'bicubic':
            return F.interpolate(lr_img, scale_factor=scale_factor, mode='bicubic', align_corners=False)
        else:
            raise ValueError(f"Unknown method: {method}")

def calculate_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
    
    return psnr

def visualize_results(image_name, lr_path, hr_path, model, device, scale_factor, save_path=None):
    lr_img = load_and_preprocess_image(lr_path)
    hr_img = load_and_preprocess_image(hr_path)
    
    # Perform super-resolution
    sr_proposed = perform_super_resolution(lr_img, model, device, 'proposed', scale_factor)
    sr_bilinear = perform_super_resolution(lr_img, model, device, 'bilinear', scale_factor)
    sr_bicubic = perform_super_resolution(lr_img, model, device, 'bicubic', scale_factor)
    
    psnr_proposed = calculate_metrics(sr_proposed, hr_img)
    psnr_bilinear = calculate_metrics(sr_bilinear, hr_img)
    psnr_bicubic = calculate_metrics(sr_bicubic, hr_img)

    # Define bottom-right ROI
    def get_bottom_right_roi(image, roi_size=60, padding=10):
        """Calculate bottom-right ROI box."""
        img_width, img_height = image.size if isinstance(image, Image.Image) else to_pil_image(image).size
        return (img_width - roi_size, img_height - roi_size, img_width - padding, img_height - padding)  # (left, top, right, bottom)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle(f'Super Resolution Comparison (x{scale_factor}) - {image_name}', fontsize=16)

    # Add zoomed-in regions
    def add_zoomed_inset(ax, img, roi_box):
        pil_img = to_pil_image(img.squeeze()) if isinstance(img, torch.Tensor) else Image.fromarray(img)
        roi = pil_img.crop(roi_box) 
        zoomed_roi = roi.resize((100, 100), Image.NEAREST) 
        
        # Draw rectangle on original image
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle(roi_box, outline="red", width=3)
        
        # Show the modified original image
        ax.imshow(pil_img)
        ax.axis('off')

        # Add the zoomed region inset
        inset_ax = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
        inset_ax.imshow(zoomed_roi)
        inset_ax.axis('off')

    # Plot images
    titles = [
        'Low Resolution',
        'High Resolution (Ground Truth)',
        f'DOSRGAN - PSNR: {psnr_proposed:.2f}dB',
        f'Bilinear - PSNR: {psnr_bilinear:.2f}dB',
        f'Bicubic - PSNR: {psnr_bicubic:.2f}dB',
        'Error Map (Amplified 5x)'
    ]

    images = [
        lr_img, hr_img, sr_proposed, sr_bilinear, sr_bicubic, 
        torch.abs(sr_proposed - hr_img).squeeze().permute(1, 2, 0).numpy() * 5 
    ]

    for ax, title, img in zip(axes.flat, titles, images):
        if title != 'Error Map (Amplified 5x)' and title != 'Low Resolution':
            pil_img = to_pil_image(img.squeeze()) if isinstance(img, torch.Tensor) else Image.fromarray(img)
            roi_box = get_bottom_right_roi(pil_img)  
            add_zoomed_inset(ax, img, roi_box)
        else:
            ax.imshow(to_pil_image(img.squeeze()) if isinstance(img, torch.Tensor) else img)
            ax.axis('off')
        ax.set_title(title, fontsize=10, y=-0.1)  
    
    plt.tight_layout(h_pad=2) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()




def main():
    scale_factor = 4
    
    # Paths
    model_path = f'proposed_model_optimized_x{scale_factor}.pth' 
    lr_dir = f'set5/x{scale_factor}/lr'
    hr_dir = f'set5/x{scale_factor}/hr'
    output_dir = f'comparison_results_x{scale_factor}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading x{scale_factor} model...")
    model, device = load_trained_model(model_path, scale_factor)
    print("\nModel loaded successfully")
    
    image_names = ['baby.png', 'head.png', 'butterfly.png', 'bird.png', 'woman.png']
    
    for img_name in image_names:
        print(f"\nProcessing {img_name}...")
        lr_path = os.path.join(lr_dir, img_name)
        hr_path = os.path.join(hr_dir, img_name)
        save_path = os.path.join(output_dir, f'comparison_{img_name.split(".")[0]}_{scale_factor}.png')
        
        visualize_results(img_name, lr_path, hr_path, model, device, scale_factor, save_path)
    
    print("\nAll images processed successfully")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
