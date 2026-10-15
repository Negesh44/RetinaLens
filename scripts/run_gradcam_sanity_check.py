"""Adebayo et al. (2018) Model Parameter Randomization Sanity Check for Grad-CAM.

Compares feature attributions between the trained clinical model and
an untrained randomly initialized model to confirm explanations depend
on learned clinical weights rather than network architecture priors.

Usage:
    python scripts/run_gradcam_sanity_check.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from ml.data import create_synthetic_fundus_image
from ml.models import create_retina_model
from ml.preprocessing import preprocess_fundus_image, get_val_transforms
from ml.gradcam import run_randomization_sanity_check


def main():
    print("=" * 70)
    print("RetinaLens: Adebayo et al. Grad-CAM Randomization Sanity Check")
    print("=" * 70)

    # 1. Create a test image
    img_rgb = create_synthetic_fundus_image(grade=3, size=224, seed=999)
    preprocessed = preprocess_fundus_image(img_rgb, target_size=224, use_ben_graham=True)
    val_tf = get_val_transforms(224)
    tensor = val_tf(image=preprocessed)["image"].unsqueeze(0)

    # 2. Instantiate trained model (or checkpoint if available)
    trained_model = create_retina_model("efficientnet_b0", pretrained=False, head_type="regression")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_regression_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        trained_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained checkpoint from: {ckpt_path.name}")
    else:
        print("Using initialized model weights for demonstration verification.")

    # 3. Run Randomization Test
    results = run_randomization_sanity_check(
        trained_model=trained_model,
        input_tensor=tensor,
        original_rgb=preprocessed,
        backbone_name="efficientnet_b0",
        head_type="regression",
    )

    print(f"\nPearson Correlation (r):       {results['pearson_correlation']:.4f}")
    print(f"Mean Absolute Difference:      {results['mean_absolute_difference']:.4f}")
    print(f"Passed Criterion (r < 0.75):   {results['passed']}")
    print(f"\nOutcome:\n{results['conclusion']}\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
