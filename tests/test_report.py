"""Unit Tests for ReportLab PDF Report Generation."""

from pathlib import Path
from datetime import datetime, timezone
import pytest

from backend.models import ScreeningRecord
from backend.report import generate_pdf_report
from ml.config import CLINICAL_DISCLAIMER


def test_pdf_report_generation(tmp_path):
    """Verifies complete PDF report generation and file structure."""
    record = ScreeningRecord(
        id="test-uuid-12345",
        created_at=datetime.now(timezone.utc),
        patient_id="PAT-UNIT-TEST",
        patient_age=62,
        eye_side="OD",
        status="completed",
        quality_score=0.92,
        quality_acceptable=True,
        grade=2,
        grade_label="Moderate",
        confidence=0.895,
        is_referable=True,
        uncertain=False,
        original_image_path=None,
        heatmap_image_path=None,
        model_version="RetinaLens-v1.0",
        disclaimer=CLINICAL_DISCLAIMER,
    )
    record.probabilities = {"0": 0.02, "1": 0.08, "2": 0.85, "3": 0.04, "4": 0.01}

    out_file = tmp_path / "test_report.pdf"
    generated_path = generate_pdf_report(record, output_path=out_file)

    assert generated_path.exists()
    assert generated_path.suffix == ".pdf"
    assert generated_path.stat().st_size > 1000

    # Validate PDF magic header bytes %PDF
    with open(generated_path, "rb") as f:
        header = f.read(4)
        assert header == b"%PDF"
