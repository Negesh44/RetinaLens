"""SQLAlchemy Database Models for RetinaLens."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text

from backend.database import Base


class ScreeningRecord(Base):
    """Database entity representing a single clinical fundus screening event."""

    __tablename__ = "screenings"

    id = Column(String(36), primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Patient Metadata
    patient_id = Column(String(64), nullable=True, index=True)
    patient_age = Column(Integer, nullable=True)
    eye_side = Column(String(16), default="Unknown")  # OD (Right Eye), OS (Left Eye), Unknown

    # Status: 'completed' or 'rejected_for_quality'
    status = Column(String(32), default="completed", index=True)

    # Image Quality Gate Results
    quality_score = Column(Float, nullable=False)
    quality_acceptable = Column(Boolean, default=True)
    quality_reasons_json = Column(Text, default="[]")

    # Neural Grading Output (ICDR 0-4)
    grade = Column(Integer, nullable=True, index=True)
    grade_label = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    probabilities_json = Column(Text, default="{}")

    # Clinical Referral & Uncertainty
    is_referable = Column(Boolean, default=False, index=True)
    uncertain = Column(Boolean, default=False)
    uncertainty_note = Column(String(255), nullable=True)
    referral_recommendation = Column(Text, nullable=True)

    # Interactive Lesion Analysis & Quantification
    lesion_count = Column(Integer, default=0)
    overall_affected_area_percentage = Column(Float, nullable=True)
    valid_retinal_pixels = Column(Integer, nullable=True)
    total_image_pixels = Column(Integer, nullable=True)
    lesions_json = Column(Text, default="[]")
    class_summaries_json = Column(Text, default="{}")

    # File Storage Paths
    original_image_path = Column(String(255), nullable=True)
    analyzed_image_path = Column(String(255), nullable=True)
    heatmap_image_path = Column(String(255), nullable=True)
    raw_heatmap_path = Column(String(255), nullable=True)
    pdf_report_path = Column(String(255), nullable=True)

    # Multi-Condition Retinal Pathology & Differential Assessment
    primary_disease = Column(String(128), nullable=True, index=True)
    disease_findings_json = Column(Text, default="[]")

    # Audit & Regulatory
    model_version = Column(String(64), default="RetinaLens-v1.0")
    disclaimer = Column(Text, nullable=False)

    @property
    def quality_reasons(self) -> List[str]:
        try:
            return json.loads(self.quality_reasons_json)
        except Exception:
            return []

    @quality_reasons.setter
    def quality_reasons(self, value: List[str]):
        self.quality_reasons_json = json.dumps(value)

    @property
    def probabilities(self) -> Dict[str, float]:
        try:
            return json.loads(self.probabilities_json)
        except Exception:
            return {}

    @probabilities.setter
    def probabilities(self, value: Dict[str, float]):
        self.probabilities_json = json.dumps(value)

    @property
    def lesions(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.lesions_json)
        except Exception:
            return []

    @lesions.setter
    def lesions(self, value: List[Dict[str, Any]]):
        self.lesions_json = json.dumps(value)

    @property
    def class_summaries(self) -> Dict[str, Any]:
        try:
            return json.loads(self.class_summaries_json)
        except Exception:
            return {}

    @class_summaries.setter
    def class_summaries(self, value: Dict[str, Any]):
        self.class_summaries_json = json.dumps(value)

    @property
    def disease_findings(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.disease_findings_json)
        except Exception:
            return []

    @disease_findings.setter
    def disease_findings(self, value: List[Dict[str, Any]]):
        self.disease_findings_json = json.dumps(value)

