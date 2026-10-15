"""Retinal Fundus Image Quality Gate.

Evaluates:
1. Sharpness/Blur via Variance of Laplacian on retinal tissue.
2. Illumination and contrast (underexposure, overexposure/flash glare, RMS contrast).
3. Field of View (FOV) retinal disk coverage ratio.

Returns an interpretable composite score [0.0, 1.0], individual subscores,
actionable clinical rejection reasons, and retake recommendations.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# Long-side resolution at which sharpness is measured.
SHARPNESS_REF_SIZE = 512


class ImageQualityAssessor:
    """Multi-factorial rule-based retinal image quality assessment engine."""

    def __init__(
        self,
        min_quality_threshold: float = 0.50,
        blur_threshold: float = 10.0,  # Minimum Laplacian variance, measured at SHARPNESS_REF_SIZE
        min_mean_intensity: float = 35.0,  # Minimum acceptable mean intensity inside fundus
        max_mean_intensity: float = 210.0,  # Maximum acceptable mean intensity (flash washout)
        min_contrast_rms: float = 25.0,  # Minimum RMS contrast
        min_fov_ratio: float = 0.45,  # Minimum circular fundus mask coverage
        max_saturation_ratio: float = 0.12,  # Max fraction of overexposed pixels
    ):
        self.min_quality_threshold = min_quality_threshold
        self.blur_threshold = blur_threshold
        self.min_mean_intensity = min_mean_intensity
        self.max_mean_intensity = max_mean_intensity
        self.min_contrast_rms = min_contrast_rms if min_contrast_rms != 25.0 else 16.0
        self.min_fov_ratio = min_fov_ratio
        self.max_saturation_ratio = max_saturation_ratio

    def assess_quality(
        self,
        image_rgb: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Union[float, bool, List[str], Dict[str, float]]]:
        """Assesses input RGB retinal image quality.

        Args:
            image_rgb: RGB uint8 numpy image.
            threshold: Optional override for acceptance threshold.

        Returns:
            Dictionary with quality_score, is_acceptable, reasons, recommendation, and metrics.
        """
        active_threshold = threshold if threshold is not None else self.min_quality_threshold
        reasons: List[str] = []

        if image_rgb is None or image_rgb.size == 0:
            return {
                "quality_score": 0.0,
                "is_acceptable": False,
                "reasons": ["Empty or corrupted image payload."],
                "recommendation": "Retake image: please verify camera capture and re-upload.",
                "metrics": {},
            }

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Retinal disc mask segmentation (distinguish fundus from camera black surround)
        fundus_mask = gray > 15
        fundus_pixels = gray[fundus_mask]
        total_pixels = h * w

        if len(fundus_pixels) == 0:
            return {
                "quality_score": 0.0,
                "is_acceptable": False,
                "reasons": ["No retinal illumination detected (completely dark image)."],
                "recommendation": "Retake image: fundus illumination flash failed or lens cap was on.",
                "metrics": {},
            }

        # 2. Field-of-View (FOV) calculation
        fov_ratio = float(len(fundus_pixels) / total_pixels)
        # Retinal disc typically occupies 45% - 85% of full rectangular sensor area
        fov_subscore = float(np.clip(fov_ratio / 0.65, 0.0, 1.0))
        if fov_ratio < self.min_fov_ratio:
            reasons.append(
                f"Inadequate Field of View: retinal disc occupies only {fov_ratio*100:.1f}% "
                f"of the frame (minimum required {self.min_fov_ratio*100:.0f}%)."
            )

        # 3. Sharpness / Blur Assessment via Laplacian Variance
        # Apply only within the fundus tissue area to avoid border edge skewing
        # Laplacian variance depends on pixel scale, so measure at a fixed resolution:
        # otherwise sharp high-resolution photos read as "blurry" and small images as sharp.
        scale = SHARPNESS_REF_SIZE / max(h, w)
        ref_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        gray_ref = cv2.resize(gray, ref_size, interpolation=cv2.INTER_AREA)
        laplacian = cv2.Laplacian(gray_ref, cv2.CV_64F)
        laplacian_fundus = laplacian[gray_ref > 15]
        laplacian_var = float(np.var(laplacian_fundus)) if len(laplacian_fundus) > 0 else 0.0

        # Soft sigmoid around the threshold maps the variance to a [0, 1] subscore
        blur_subscore = float(1.0 / (1.0 + np.exp(-0.12 * (laplacian_var - self.blur_threshold))))
        if laplacian_var < self.blur_threshold:
            reasons.append(
                f"Significant blur / optical defocus detected (sharpness metric {laplacian_var:.1f} "
                f"< threshold {self.blur_threshold:.1f})."
            )

        # 4. Illumination & Contrast Assessment
        mean_intensity = float(np.mean(fundus_pixels))
        contrast_rms = float(np.std(fundus_pixels))
        saturated_pixels = np.sum(fundus_pixels > 240)
        saturation_ratio = float(saturated_pixels / len(fundus_pixels))

        # Contrast subscore
        contrast_subscore = float(np.clip(contrast_rms / 40.0, 0.0, 1.0))

        # Illumination subscore
        if mean_intensity < self.min_mean_intensity:
            illum_subscore = float(np.clip(mean_intensity / self.min_mean_intensity, 0.0, 1.0))
            reasons.append(
                f"Severely underexposed / dark fundus (mean illumination {mean_intensity:.1f} "
                f"< minimum {self.min_mean_intensity:.1f})."
            )
        elif mean_intensity > self.max_mean_intensity:
            illum_subscore = float(np.clip((255.0 - mean_intensity) / (255.0 - self.max_mean_intensity), 0.0, 1.0))
            reasons.append(
                f"Overexposed capture / excessive illumination (mean {mean_intensity:.1f} > maximum {self.max_mean_intensity:.1f})."
            )
        else:
            illum_subscore = 1.0

        if saturation_ratio > self.max_saturation_ratio:
            reasons.append(
                f"Corneal flash glare / saturation artifact detected ({saturation_ratio*100:.1f}% "
                f"of retinal pixels saturated > threshold {self.max_saturation_ratio*100:.0f}%)."
            )
            illum_subscore *= 0.7

        if contrast_rms < self.min_contrast_rms:
            reasons.append(
                f"Low retinal contrast (RMS contrast {contrast_rms:.1f} < threshold {self.min_contrast_rms:.1f}); "
                "may indicate media opacity or cataract."
            )

        # 5. Composite Weighted Quality Score
        composite_score = float(
            0.40 * blur_subscore +
            0.30 * illum_subscore +
            0.15 * contrast_subscore +
            0.15 * fov_subscore
        )
        composite_score = round(float(np.clip(composite_score, 0.0, 1.0)), 3)

        # Hard failure on severe blur, severe darkness, or inadequate FOV
        has_hard_defect = (
            laplacian_var < self.blur_threshold or
            mean_intensity < self.min_mean_intensity or
            mean_intensity > self.max_mean_intensity or
            fov_ratio < self.min_fov_ratio
        )

        if has_hard_defect:
            composite_score = min(composite_score, round(active_threshold - 0.05, 3))

        is_acceptable = (composite_score >= active_threshold) and not has_hard_defect

        if is_acceptable:
            recommendation = "Image quality is acceptable for clinical AI screening."
        else:
            recommendation = (
                "RETAKE IMAGE: Fundus capture does not meet diagnostic quality criteria. "
                "Please re-align the fundus camera, adjust illumination and focus, and capture again."
            )

        return {
            "quality_score": composite_score,
            "is_acceptable": is_acceptable,
            "reasons": reasons,
            "recommendation": recommendation,
            "metrics": {
                "sharpness_laplacian_var": round(laplacian_var, 2),
                "mean_intensity": round(mean_intensity, 2),
                "contrast_rms": round(contrast_rms, 2),
                "fov_coverage_ratio": round(fov_ratio, 3),
                "saturation_ratio": round(saturation_ratio, 3),
                "blur_subscore": round(blur_subscore, 3),
                "illumination_subscore": round(illum_subscore, 3),
                "contrast_subscore": round(contrast_subscore, 3),
                "fov_subscore": round(fov_subscore, 3),
            },
        }


# Global assessor instance for fast invocation
quality_assessor = ImageQualityAssessor()
