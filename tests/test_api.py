"""Unit & Integration Tests for RetinaLens FastAPI Backend."""

import io
import cv2
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db
from ml.data import create_synthetic_fundus_image
from ml.config import CLINICAL_DISCLAIMER


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "RetinaLens"
    assert data["disclaimer"] == CLINICAL_DISCLAIMER


def test_api_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_screenings" in data
    assert "referral_rate_pct" in data
    assert "grade_distribution" in data
    assert "daily_trends" in data


def test_api_screen_accepts_and_grades(client):
    """Submits a clear synthetic fundus image to POST /api/screen."""
    img = create_synthetic_fundus_image(grade=0, size=250, seed=777)
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_bytes = buf.tobytes()

    response = client.post(
        "/api/screen",
        files={"file": ("test_retina.png", img_bytes, "image/png")},
        data={"patient_id": "TEST-PAT-01", "patient_age": "54", "eye_side": "OD"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert data["patient_id"] == "TEST-PAT-01"
    assert data["eye_side"] == "OD"
    assert data["quality_acceptable"] is True
    assert data["quality_score"] >= 0.50
    assert data["grade"] in [0, 1, 2, 3, 4]
    assert data["grade_label"] is not None
    assert data["confidence"] is not None
    assert "disclaimer" in data
    assert data["disclaimer"] == CLINICAL_DISCLAIMER
    assert data["original_image_url"] is not None
    assert data["heatmap_url"] is not None


def test_api_screen_rejects_poor_quality(client):
    """Submits a heavily blurred image to verify Quality Gate refuses to grade."""
    img = create_synthetic_fundus_image(grade=2, size=250, seed=888)
    blurred = cv2.GaussianBlur(img, (41, 41), 18.0)
    _, buf = cv2.imencode(".png", cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
    img_bytes = buf.tobytes()

    response = client.post(
        "/api/screen",
        files={"file": ("blurred_retina.png", img_bytes, "image/png")},
        data={"patient_id": "TEST-REJECT-01", "eye_side": "OS"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "rejected_for_quality"
    assert data["quality_acceptable"] is False
    assert data["quality_score"] < 0.50
    assert len(data["quality_reasons"]) > 0
    assert "retake" in data["referral_recommendation"].lower()
    assert data["grade"] is None
    assert data["disclaimer"] == CLINICAL_DISCLAIMER


def test_api_results_list(client):
    response = client.get("/api/results?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
