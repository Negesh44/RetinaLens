"""Unit & Integration Tests for Retinal Lesion Detection, Segmentation, and Area Calculations."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db
from ml.data import create_synthetic_fundus_image
from ml.lesion_detector import (
    RetinaLesionDetector,
    extract_valid_retina_mask,
    determine_retinal_location,
    LESION_COLORS,
)


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client


def test_valid_retina_mask_excludes_borders():
    """Valid retinal field mask must accurately isolate the fundus circle and discard black borders."""
    img = create_synthetic_fundus_image(grade=0, size=300, seed=101)
    valid_mask, valid_pixels = extract_valid_retina_mask(img)

    total_pixels = 300 * 300
    assert valid_mask.shape == (300, 300)
    assert valid_pixels > 0
    # Fundus circle radius is ~44%, area is roughly pi * (0.44 * 300)^2 ~ 54,000 px out of 90,000 px (~60%)
    coverage_ratio = valid_pixels / total_pixels
    assert 0.40 <= coverage_ratio <= 0.75, f"Unexpected coverage ratio: {coverage_ratio}"

    # Corner pixel at (0, 0) must be False (background border)
    assert valid_mask[0, 0] == False
    assert valid_mask[299, 299] == False
    # Center pixel at (150, 150) must be True
    assert valid_mask[150, 150] == True


def test_affected_area_percentage_formula():
    """Affected Area (%) = (Number of lesion pixels / Number of valid retinal pixels) * 100."""
    detector = RetinaLesionDetector()
    img = create_synthetic_fundus_image(grade=2, size=300, seed=202)
    result = detector.detect_lesions(img)

    assert result.analysis_status == "completed"
    assert result.valid_retinal_pixels > 0

    # Test each individual lesion's formula
    for lesion in result.lesions:
        expected_pct = (lesion.area_pixels / result.valid_retinal_pixels) * 100.0
        assert np.isclose(lesion.affected_area_percentage, expected_pct, atol=1e-3)
        assert lesion.affected_area_percentage >= 0.0
        assert lesion.affected_area_percentage <= 100.0
        assert lesion.confidence is not None
        assert lesion.confidence >= 0.65


def test_union_mask_avoids_double_counting():
    """Total affected area percentage must be calculated via union of masks, avoiding double-counting."""
    detector = RetinaLesionDetector()
    img = create_synthetic_fundus_image(grade=3, size=300, seed=303)
    result = detector.detect_lesions(img)

    # Sum of individual lesion percentages
    individual_pct_sum = sum(l.affected_area_percentage for l in result.lesions)
    # Class percentages sum
    class_pct_sum = sum(cs["affected_area_percentage"] for cs in result.class_summaries.values())

    # Overall union percentage must be less than or equal to simple linear sum
    # (due to no double counting on overlapping boundary pixels)
    assert result.overall_affected_area_percentage <= (individual_pct_sum + 1e-4)
    assert result.overall_affected_area_percentage <= (class_pct_sum + 1e-4)


def test_coordinate_normalization_and_quadrant_mapping():
    """Coordinates and bounding boxes must be properly bounded in [0.0, 1.0]."""
    detector = RetinaLesionDetector()
    img = create_synthetic_fundus_image(grade=2, size=300, seed=404)
    result = detector.detect_lesions(img)

    assert len(result.lesions) > 0
    for lesion in result.lesions:
        # Bounding box [x_min, y_min, x_max, y_max]
        bx1, by1, bx2, by2 = lesion.bbox
        assert 0 <= bx1 <= bx2 <= 300
        assert 0 <= by1 <= by2 <= 300

        # Normalized bbox [nx1, ny1, nx2, ny2]
        nbx1, nby1, nbx2, nby2 = lesion.normalized_bbox
        assert 0.0 <= nbx1 <= nbx2 <= 1.0
        assert 0.0 <= nby1 <= nby2 <= 1.0

        # Normalized centroid
        ncx, ncy = lesion.normalized_centroid
        assert 0.0 <= ncx <= 1.0
        assert 0.0 <= ncy <= 1.0

        # Location string must specify quadrant or macular region
        assert any(term in lesion.location for term in ["Upper", "Lower", "Central", "Macular"])

        # Polygon
        assert len(lesion.polygon) >= 3
        for pt in lesion.normalized_polygon:
            assert 0.0 <= pt[0] <= 1.0
            assert 0.0 <= pt[1] <= 1.0


def test_api_analyze_retina_endpoint_structure(client):
    """POST /api/analyze-retina must return structured JSON matching specification."""
    img = create_synthetic_fundus_image(grade=2, size=300, seed=505)
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_bytes = buf.tobytes()

    response = client.post(
        "/api/analyze-retina",
        files={"file": ("fundus_exam.png", img_bytes, "image/png")},
        data={"patient_id": "PAT-AI-TEST", "eye_side": "OD"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["analysis_status"] == "completed"
    assert "image_id" in data
    assert "retinal_grade" in data
    assert "lesion_count" in data
    assert data["lesion_count"] > 0
    assert "overall_affected_area_percentage" in data
    if any(f.get("model_name") for f in data.get("disease_findings", [])):
        # Trained classifier: lesions are AI attention areas, which have no measured lesion area.
        assert data["overall_affected_area_percentage"] is None
        assert all(lesion["class_name"] == "attention_region" for lesion in data["lesions"])
    else:
        assert data["overall_affected_area_percentage"] > 0.0
    assert "calculation_method" in data
    assert "class_summaries" in data
    assert "lesions" in data
    assert len(data["lesions"]) == data["lesion_count"]

    first_lesion = data["lesions"][0]
    required_keys = [
        "id", "class_name", "display_name", "confidence", "affected_area_percentage",
        "area_pixels", "bbox", "normalized_bbox", "polygon", "normalized_polygon",
        "normalized_centroid", "location", "explanation", "severity", "educational_advice"
    ]
    for k in required_keys:
        assert k in first_lesion, f"Missing key {k} in lesion DTO"

    assert data["analyzed_image_url"] is not None


def test_api_quality_rejection_does_not_hallucinate_lesions(client):
    """Heavily degraded image must be rejected by quality gate without hallucinating lesions."""
    img = create_synthetic_fundus_image(grade=2, size=300, seed=606)
    blurred = cv2.GaussianBlur(img, (45, 45), 20.0)
    _, buf = cv2.imencode(".png", cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))

    response = client.post(
        "/api/analyze-retina",
        files={"file": ("blurred_sample.png", buf.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["analysis_status"] == "rejected_for_quality"
    assert data["quality_acceptable"] is False
    assert data["lesion_count"] == 0
    assert data["overall_affected_area_percentage"] is None
    assert len(data["lesions"]) == 0
