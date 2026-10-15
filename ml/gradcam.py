"""Explainable AI (XAI) Engine for Retinal Fundus Screening.

Provides:
1. Grad-CAM and Grad-CAM++ feature attribution on the final convolutional layer.
2. Raw normalized heatmaps and colormapped overlay generation (TURBO / JET).
3. Model Parameter Randomization Sanity Check (Adebayo et al., NeurIPS 2018)
   to confirm explanations depend on learned clinical weights.
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from ml.models import RetinaModel, create_retina_model


class ModelScalarTarget:
    """Target functor for continuous regression scalar predictions in Grad-CAM."""

    def __init__(self, target_category: Optional[int] = None):
        self.target_category = target_category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[0]
        elif model_output.ndim == 2:
            if self.target_category is not None and model_output.shape[1] > self.target_category:
                return model_output[0, self.target_category]
            return model_output[0, 0]
        return model_output.squeeze()


class FundusExplainer:
    """XAI manager providing Grad-CAM and Grad-CAM++ visualizations."""

    def __init__(
        self,
        model: nn.Module,
        method: str = "gradcam",  # 'gradcam' or 'gradcam++'
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.method_name = method.lower()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

        # Locate target convolutional layer
        if hasattr(self.model, "get_target_layer_for_cam"):
            self.target_layer = self.model.get_target_layer_for_cam()
        else:
            # Fallback search for last Conv2d
            conv_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
            if not conv_layers:
                raise RuntimeError("No Conv2d layers detected in model architecture for Grad-CAM.")
            self.target_layer = conv_layers[-1]

        cam_class = GradCAMPlusPlus if "++" in self.method_name else GradCAM
        self.cam = cam_class(model=self.model, target_layers=[self.target_layer])

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        target_category: Optional[int] = None,
        colormap: int = cv2.COLORMAP_TURBO,
        blend_alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates raw heatmap and blended overlay.

        Args:
            input_tensor: Tensor of shape [1, 3, H, W] normalized with ImageNet stats.
            original_rgb: RGB uint8 numpy image of shape [H, W, 3] (0-255).
            target_category: Optional class index target.
            colormap: OpenCV colormap (cv2.COLORMAP_TURBO default for high clinical contrast).
            blend_alpha: Alpha blending ratio.

        Returns:
            Tuple of:
            - raw_heatmap: Float32 array [H, W] normalized in [0.0, 1.0].
            - overlay_rgb: RGB uint8 array [H, W, 3] of blended visualization.
        """
        input_tensor = input_tensor.to(self.device)
        target = [ModelScalarTarget(target_category)]

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=target)
        raw_heatmap = grayscale_cam[0, :]  # [H, W] in [0, 1]

        # Resize heatmap to match original RGB image dimensions
        h, w = original_rgb.shape[:2]
        if raw_heatmap.shape != (h, w):
            raw_heatmap = cv2.resize(raw_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize fundus image to [0, 1] float
        norm_img = original_rgb.astype(np.float32) / 255.0
        norm_img = np.clip(norm_img, 0.0, 1.0)

        # Generate smooth overlay
        overlay_rgb = show_cam_on_image(
            norm_img,
            raw_heatmap,
            use_rgb=True,
            colormap=colormap,
            image_weight=1.0 - blend_alpha,
        )
        return raw_heatmap, overlay_rgb


def run_randomization_sanity_check(
    trained_model: RetinaModel,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    backbone_name: str = "efficientnet_b0",
    head_type: str = "regression",
) -> Dict[str, Union[float, bool, str]]:
    """Adebayo et al.

    (NeurIPS 2018) Parameter Randomization Sanity Check.
    Compares Grad-CAM explanation of trained model against an identical randomly initialized model.
    """
    explainer_trained = FundusExplainer(trained_model, method="gradcam")
    raw_trained, _ = explainer_trained.explain(input_tensor, original_rgb)

    # Instantiate identical untrained model with randomized weights
    random_model = create_retina_model(
        backbone_name=backbone_name,
        pretrained=False,
        head_type=head_type,
    )
    explainer_random = FundusExplainer(random_model, method="gradcam")
    raw_random, _ = explainer_random.explain(input_tensor, original_rgb)

    # Compute Pearson correlation coefficient between heatmaps
    flat_trained = raw_trained.flatten()
    flat_random = raw_random.flatten()

    std_trained = np.std(flat_trained)
    std_random = np.std(flat_random)

    if std_trained > 1e-6 and std_random > 1e-6:
        pearson_r = float(np.corrcoef(flat_trained, flat_random)[0, 1])
    else:
        pearson_r = 0.0

    # Mean Absolute Difference
    mad = float(np.mean(np.abs(raw_trained - raw_random)))

    # A genuine explanation will diverge from the untrained model (pearson_r significantly < 0.8)
    passed = bool(pearson_r < 0.75)

    return {
        "passed": passed,
        "pearson_correlation": round(pearson_r, 4),
        "mean_absolute_difference": round(mad, 4),
        "conclusion": (
            "SANITY CHECK PASSED: Explanation significantly changed upon weight randomization, "
            f"demonstrating dependence on learned clinical parameters (r = {pearson_r:.3f})."
            if passed else
            f"SANITY CHECK WARNING: High correlation with random model (r = {pearson_r:.3f}); "
            "heatmap may reflect architectural edge priors."
        ),
    }
