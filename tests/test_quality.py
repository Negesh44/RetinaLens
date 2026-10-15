"""Unit Tests for Image Quality Gate."""

import cv2
import numpy as np
import pytest
from ml.quality import ImageQualityAssessor
from ml.data import create_synthetic_fundus_image


@pytest.fixture
def assessor():
    return ImageQualityAssessor(min_quality_threshold=0.50)


def test_quality_gate_accepts_good_image(assessor):
    """A standard crisp retinal fundus should pass the quality check."""
    good_img = create_synthetic_fundus_image(grade=0, size=300, seed=42)
    result = assessor.assess_quality(good_img)

    assert result["is_acceptable"] is True
    assert result["quality_score"] >= 0.50
    assert len(result["reasons"]) == 0
    assert "acceptable" in result["recommendation"].lower()


def test_quality_gate_rejects_heavy_blur(assessor):
    """Heavily defocused image should be intercepted with specific blur explanation."""
    base_img = create_synthetic_fundus_image(grade=2, size=300, seed=42)
    # Heavy Gaussian Blur
    blurred = cv2.GaussianBlur(base_img, (35, 35), 15.0)

    result = assessor.assess_quality(blurred)
    assert result["is_acceptable"] is False
    assert result["quality_score"] < 0.50
    assert any("blur" in r.lower() or "defocus" in r.lower() for r in result["reasons"])
    assert "retake" in result["recommendation"].lower()


def test_quality_gate_rejects_underexposed_image(assessor):
    """Severe underexposure should be flagged and rejected."""
    base_img = create_synthetic_fundus_image(grade=1, size=300, seed=42)
    dark_img = (base_img * 0.10).astype(np.uint8)

    result = assessor.assess_quality(dark_img)
    assert result["is_acceptable"] is False
    assert any("underexposed" in r.lower() or "dark" in r.lower() or "contrast" in r.lower() for r in result["reasons"])


def test_quality_gate_handles_empty_image(assessor):
    """Empty numpy array should not crash and return 0.0 score."""
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = assessor.assess_quality(empty_img)
    assert result["is_acceptable"] is False
    assert result["quality_score"] == 0.0
