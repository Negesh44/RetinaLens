"""Unit Tests for Image Preprocessing and Ben Graham Contrast Enhancement."""

import numpy as np
import pytest
from ml.preprocessing import (
    crop_retina_border,
    apply_ben_graham_enhancement,
    preprocess_fundus_image,
)
from ml.data import create_synthetic_fundus_image


def test_crop_retina_border():
    """Verify black border removal on a synthetic fundus disc with padding."""
    fundus = create_synthetic_fundus_image(grade=0, size=200, seed=42)
    # Pad with 50 pixels of pure black
    padded = np.pad(fundus, ((50, 50), (50, 50), (0, 0)), mode="constant", constant_values=0)
    assert padded.shape == (300, 300, 3)

    cropped = crop_retina_border(padded, tol=10)
    # The cropped image height and width should be significantly reduced towards original 200
    assert cropped.shape[0] < 300
    assert cropped.shape[1] < 300
    assert cropped.shape[0] >= 160
    assert cropped.shape[1] >= 160


def test_apply_ben_graham_enhancement():
    """Verify local contrast normalization produces proper shape and normalized range."""
    img = create_synthetic_fundus_image(grade=2, size=224, seed=123)
    enhanced = apply_ben_graham_enhancement(img)

    assert enhanced.shape == (224, 224, 3)
    assert enhanced.dtype == np.uint8
    # Retinal center should be enhanced with reasonable contrast
    assert np.mean(enhanced) > 20.0
    assert np.std(enhanced) > 10.0


def test_preprocess_fundus_image():
    """Verify full preprocessing pipeline to target resolution."""
    raw_img = create_synthetic_fundus_image(grade=1, size=350, seed=99)
    target_dim = 224
    processed = preprocess_fundus_image(raw_img, target_size=target_dim, use_ben_graham=True)

    assert processed.shape == (target_dim, target_dim, 3)
    assert processed.dtype == np.uint8
