"""Retinal Fundus Image Preprocessing & Augmentation Pipeline.

Implements clinical-grade preprocessing:
1. Black border removal / circular retinal mask cropping.
2. Ben Graham's local contrast enhancement (Gaussian difference).
3. Circular boundary masking to eliminate peripheral halo artifacts.
4. Deterministic validation pipeline and Albumentations augmentations for training.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def crop_retina_border(
    image: np.ndarray,
    tol: int = 10,
    min_area_ratio: float = 0.25
) -> np.ndarray:
    """Crops uninformative black borders around retinal fundus images.

    Args:
        image: RGB or BGR numpy array [H, W, 3].
        tol: Grayscale intensity threshold for non-background pixels.
        min_area_ratio: Minimum ratio of cropped bbox area to original area
                        to avoid degenerate crops.

    Returns:
        Cropped numpy array image.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image

    # Threshold mask
    mask = gray > tol

    # Check if image has sufficient non-zero pixels
    if not np.any(mask):
        return image

    # Row and col indices containing signal
    row_idx = np.where(mask.any(axis=1))[0]
    col_idx = np.where(mask.any(axis=0))[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        return image

    y_min, y_max = row_idx[0], row_idx[-1] + 1
    x_min, x_max = col_idx[0], col_idx[-1] + 1

    h, w = image.shape[:2]
    crop_area = (y_max - y_min) * (x_max - x_min)
    total_area = h * w

    if (crop_area / total_area) < min_area_ratio:
        # Fall back to original image if crop is degenerate
        return image

    return image[y_min:y_max, x_min:x_max]


def apply_ben_graham_enhancement(
    image: np.ndarray,
    sigma_factor: float = 30.0,
    weight: float = 4.0
) -> np.ndarray:
    """Applies Ben Graham's local contrast enhancement (Kaggle DR 1st place technique).

    Formula: 4 * img - 4 * GaussianBlur(img, sigma) + 128
    Combined with circular mask to eliminate harsh boundary artifacts.

    Args:
        image: RGB numpy uint8 array [H, W, 3].
        sigma_factor: Factor for sigma = image_width / sigma_factor.
        weight: Multiplier weight for difference (default 4.0).

    Returns:
        Enhanced RGB image as uint8 numpy array.
    """
    h, w = image.shape[:2]
    sigma = max(int(w / sigma_factor), 10)

    # Gaussian blur for local illumination estimate
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Add weighted difference and offset to mid-gray 128
    enhanced = cv2.addWeighted(image, weight, blurred, -weight, 128)

    # Construct circular mask around fundus center to remove peripheral glare
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.49)
    cv2.circle(mask, center, radius, 255, -1)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Blend: inside circle is enhanced, outside retains gentle border fade
    result = np.where(mask_3c == 255, enhanced, cv2.GaussianBlur(image, (21, 21), 0))
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_fundus_image(
    image: np.ndarray,
    target_size: int = 224,
    use_ben_graham: bool = True,
    crop_border: bool = True
) -> np.ndarray:
    """Full preprocessing pipeline from raw fundus to standardized model input format.

    Args:
        image: Input RGB numpy array.
        target_size: Target square dimension (e.g. 224 or 380).
        use_ben_graham: Whether to apply local contrast normalization.
        crop_border: Whether to detect and crop black borders.

    Returns:
        Preprocessed RGB numpy array of shape [target_size, target_size, 3].
    """
    processed = image.copy()

    if crop_border:
        processed = crop_retina_border(processed)

    # Resize with inter-area interpolation to reduce aliasing
    processed = cv2.resize(processed, (target_size, target_size), interpolation=cv2.INTER_AREA)

    if use_ben_graham:
        processed = apply_ben_graham_enhancement(processed)

    return processed


def get_train_transforms(image_size: int = 224) -> "A.Compose":
    """Albumentations training augmentation pipeline with standard clinical variations."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=180,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6,
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.05,
            p=0.5,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> "A.Compose":
    """Deterministic validation and inference transform pipeline."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_model_input(image_rgb: np.ndarray) -> np.ndarray:
    """NumPy equivalent of get_val_transforms: ImageNet-normalised float32 array [1, 3, H, W]."""
    x = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
