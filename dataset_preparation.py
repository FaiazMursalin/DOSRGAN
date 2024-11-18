import os
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import cv2
from tqdm import tqdm


def create_patches(img, patch_size=96, stride=48):
    """Extract patches from image with augmentation"""
    patches = []
    w, h = img.size

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)
            patches.append(patch.transpose(Image.FLIP_LEFT_RIGHT))
            patches.append(patch.transpose(Image.FLIP_TOP_BOTTOM))
            patches.append(patch.rotate(90))

    return patches


def create_dataset_structure(base_path, dataset_name, scale_factors):
    """Create directory structure for the dataset."""
    dataset_path = Path(base_path) / dataset_name
    print(f"Creating directory structure at: {dataset_path}")

    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
        for scale in scale_factors:
            scale_path = dataset_path / f'x{scale}'
            hr_path = scale_path / 'hr'
            lr_path = scale_path / 'lr'
            hr_path.mkdir(parents=True, exist_ok=True)
            lr_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directories for scale {scale}")
    except Exception as e:
        print(f"Error creating directory structure: {str(e)}")
        raise


def process_image(img_path, output_dir, scale_factor, patch_size=96, stride=48, is_training=True):
    """Process a single image and save its HR and LR patches."""
    try:
        print(f"\nProcessing image: {img_path}")
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        min_size = patch_size * 2

        if w < min_size or h < min_size:
            print(f"Skipping {img_path} - image too small ({w}x{h} < {min_size}x{min_size})")
            return

        if is_training:
            hr_patches = create_patches(img, patch_size, stride)

            for idx, hr_patch in enumerate(hr_patches):
                w, h = hr_patch.size
                lr_size = (w // scale_factor, h // scale_factor)

                hr_np = np.array(hr_patch)
                blurred = cv2.GaussianBlur(hr_np, (5, 5), 0.5)
                hr_blurred = Image.fromarray(blurred)
                lr_patch = hr_blurred.resize(lr_size, Image.BICUBIC)

                lr_np = np.array(lr_patch)
                if lr_np.std() < 10:
                    continue

                img_name = Path(img_path).stem
                hr_path = output_dir / f'x{scale_factor}/hr/{img_name}_patch_{idx}.png'
                lr_path = output_dir / f'x{scale_factor}/lr/{img_name}_patch_{idx}.png'

                hr_patch.save(hr_path, 'PNG', quality=100)
                lr_patch.save(lr_path, 'PNG')
        else:
            w, h = img.size
            new_w = ((w + scale_factor - 1) // scale_factor) * scale_factor
            new_h = ((h + scale_factor - 1) // scale_factor) * scale_factor

            if new_w != w or new_h != h:
                padding = ((new_w - w) // 2, (new_h - h) // 2,
                           (new_w - w + 1) // 2, (new_h - h + 1) // 2)
                img = ImageOps.expand(img, padding, fill=0)

            img_np = np.array(img)
            blurred = cv2.GaussianBlur(img_np, (5, 5), 0.5)
            img_blurred = Image.fromarray(blurred)

            img_name = Path(img_path).stem
            hr_path = output_dir / f'x{scale_factor}/hr/{img_name}.png'
            lr_path = output_dir / f'x{scale_factor}/lr/{img_name}.png'

            img.save(hr_path, 'PNG', quality=100)

            lr_size = (new_w // scale_factor, new_h // scale_factor)
            lr_img = img_blurred.resize(lr_size, Image.BICUBIC)
            lr_img.save(lr_path, 'PNG')

    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        raise


def prepare_dataset(source_dir, output_base_dir, dataset_name, scale_factors=[2, 3, 4],
                    patch_size=96, stride=48, is_training=True):
    """Prepare the complete dataset with improved patch extraction and processing."""
    try:
        print(f"\nPreparing dataset: {dataset_name}")
        print(f"Source directory: {source_dir}")
        print(f"Output directory: {output_base_dir}")
        print(f"Patch size: {patch_size}, Stride: {stride}")

        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        create_dataset_structure(output_base_dir, dataset_name, scale_factors)

        source_path = Path(source_dir)
        output_dir = Path(output_base_dir) / dataset_name

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in source_path.glob('*')
                       if f.suffix.lower() in valid_extensions]
        total_files = len(image_files)
        print(f"Found {total_files} valid images in source directory")

        if total_files == 0:
            print("Warning: No valid images found in source directory!")
            return

        for img_path in tqdm(image_files, desc="Processing images"):
            for scale_factor in scale_factors:
                process_image(
                    img_path, output_dir, scale_factor,
                    patch_size, stride, is_training
                )

        print(f"\nDataset preparation completed for {dataset_name}")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise


if __name__ == '__main__':
    # Configuration
    scale_factors = [2, 3, 4]
    base_output_dir = './dataset_improved'
    patch_size = 96
    stride = 48

    try:
        # Prepare training dataset (General100)
        prepare_dataset(
            source_dir='data/General100',
            output_base_dir=base_output_dir,
            dataset_name='general100',
            scale_factors=scale_factors,
            patch_size=patch_size,
            stride=stride,
            is_training=True
        )

        # Prepare evaluation dataset (Set5)
        prepare_dataset(
            source_dir='data/Set5/original',
            output_base_dir=base_output_dir,
            dataset_name='set5',
            scale_factors=scale_factors,
            is_training=False
        )

        print("\nDataset preparation completed successfully!")

    except Exception as e:
        print(f"\nScript failed with error: {str(e)}")