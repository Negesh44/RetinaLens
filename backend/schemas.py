"""Pydantic Data Transfer Objects and Request/Response Schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ScreeningBase(BaseModel):
    patient_id: Optional[str] = Field(None, description="Clinical patient identifier")
    patient_age: Optional[int] = Field(None, ge=0, le=130, description="Patient age in years")
    eye_side: str = Field("Unknown", description="OD (Right Eye) or OS (Left Eye)")


class QualityAssessmentDTO(BaseModel):
    quality_score: float = Field(..., ge=0.0, le=1.0)
    quality_acceptable: bool
    reasons: List[str] = []
    recommendation: str


class LesionDTO(BaseModel):
    id: str = Field(..., description="Unique lesion identifier, e.g. lesion-1")
    class_name: str = Field(..., description="Lesion class: microaneurysm, hemorrhage, hard_exudate, cotton_wool_spot")
    display_name: str = Field(..., description="Human-readable finding name")
    confidence: Optional[float] = Field(None, description="Model confidence score for finding")
    affected_area_percentage: Optional[float] = Field(None, description="Percentage of valid retina covered")
    area_pixels: Optional[int] = Field(None, description="Raw pixel count of lesion")
    bbox: Optional[List[int]] = Field(None, description="[x_min, y_min, x_max, y_max] in pixels")
    normalized_bbox: Optional[List[float]] = Field(None, description="[nx_min, ny_min, nx_max, ny_max] in [0.0, 1.0]")
    polygon: Optional[List[List[int]]] = Field(None, description="List of [x, y] polygon coordinates")
    normalized_polygon: Optional[List[List[float]]] = Field(None, description="List of [nx, ny] in [0.0, 1.0]")
    segmentation_mask: Optional[List[List[int]]] = Field(None, description="Alias/representation of contour polygon")
    normalized_centroid: Optional[List[float]] = Field(None, description="[ncx, ncy] normalized coordinates")
    centroid_pixels: Optional[List[int]] = Field(None, description="[cx, cy] pixel coordinates")
    location: Optional[str] = Field(None, description="Retinal quadrant, e.g. Upper-Right (Superior-Temporal)")
    explanation: Optional[str] = Field(None, description="Clinical explanation of finding")
    severity: Optional[str] = Field(None, description="Visual severity classification (Mild/Moderate/Severe)")
    educational_advice: Optional[str] = Field(None, description="Recommended clinical next steps")
    associated_disease: Optional[str] = Field("diabetic_retinopathy", description="Associated primary retinal condition")


class DiseaseFindingDTO(BaseModel):
    disease_id: str = Field(..., description="Unique condition ID (e.g. amd, glaucoma, rvo)")
    disease_name: str = Field(..., description="Full clinical disease name")
    acronym: str = Field(..., description="Clinical shorthand acronym (e.g. AMD, RVO, CSC)")
    category: str = Field(..., description="Pathology category (e.g. Vascular, Macular, Optic Nerve)")
    is_detected: bool = Field(False, description="Whether hallmark criteria were detected")
    confidence: float = Field(0.0, description="Differential diagnostic confidence score")
    severity: str = Field("None", description="Assessed visual severity: None, Mild, Moderate, Severe, Critical")
    hallmark_findings: List[str] = Field(default_factory=list, description="Clinical hallmarks observed")
    triage_urgency: str = Field("Routine", description="Clinical urgency tier")
    clinical_action: str = Field("", description="Recommended clinical intervention")
    key_biomarkers: List[str] = Field(default_factory=list, description="Present biomarker IDs")
    model_threshold: Optional[float] = Field(None, description="Decision threshold tuned on validation data")
    test_auc: Optional[float] = Field(None, description="Held-out test ROC AUC for this condition")
    test_sensitivity: Optional[float] = Field(None, description="Held-out test sensitivity at the threshold")
    test_specificity: Optional[float] = Field(None, description="Held-out test specificity at the threshold")
    test_positives: int = Field(0, description="Number of positive cases in the held-out test set")
    model_name: str = Field("", description="Classifier that produced this finding")


class ClassSummaryDTO(BaseModel):
    count: int = Field(0, description="Count of detected lesions of this class")
    area_pixels: int = Field(0, description="Total pixels covered by this lesion class")
    affected_area_percentage: float = Field(0.0, description="Percentage of valid retina covered")



class ScreeningResponse(BaseModel):
    id: str
    image_id: Optional[str] = None  # Alias for standard schema compatibility
    created_at: datetime
    patient_id: Optional[str] = None
    patient_age: Optional[int] = None
    eye_side: str = "Unknown"
    status: str  # 'completed' or 'rejected_for_quality'
    analysis_status: Optional[str] = None  # Standardized alias

    # Quality Gate
    quality_score: float
    quality_acceptable: bool
    quality_reasons: List[str] = []

    # AI Prediction (ICDR 0-4)
    grade: Optional[int] = None
    retinal_grade: Optional[int] = None  # Standardized alias
    grade_label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = {}

    # Clinical Referral & Uncertainty
    is_referable: bool = False
    uncertain: bool = False
    uncertainty_note: Optional[str] = None
    referral_recommendation: Optional[str] = None

    # Interactive Lesion Analysis & Quantified Affected Area
    lesion_count: int = 0
    overall_affected_area_percentage: Optional[float] = None
    valid_retinal_pixels: Optional[int] = None
    total_image_pixels: Optional[int] = None
    calculation_method: Optional[str] = "Exact pixel-level union mask over segmented valid retinal field"
    image_resolution: Optional[List[int]] = None
    class_summaries: Dict[str, ClassSummaryDTO] = {}
    lesions: List[LesionDTO] = []
    warnings: List[str] = []

    # Multi-Condition Retinal Pathology & Differential Diagnosis
    primary_disease: Optional[str] = None
    primary_disease_name: Optional[str] = None
    detected_pathologies: List[str] = []
    disease_findings: List[DiseaseFindingDTO] = []
    cup_to_disc_ratio: Optional[float] = None

    # Media URLs
    original_image_url: Optional[str] = None
    analyzed_image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    report_download_url: Optional[str] = None

    # Metadata & Disclaimer
    model_version: str
    disclaimer: str

    class Config:
        from_attributes = True


class ScreeningListResponse(BaseModel):
    total: int
    items: List[ScreeningResponse]


class DailyScreeningCount(BaseModel):
    date: str
    total: int
    referable: int
    rejected: int


class DashboardStatsResponse(BaseModel):
    total_screenings: int
    completed_screenings: int
    rejected_for_quality_count: int
    referable_count: int
    referral_rate_pct: float
    mean_confidence: float
    grade_distribution: Dict[int, int]
    grade_labels: Dict[int, str]
    daily_trends: List[DailyScreeningCount]


class ChatMessageDTO(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    screening_id: Optional[str] = Field(None, description="Screening whose report grounds the answer")
    messages: List[ChatMessageDTO] = Field(..., min_length=1, max_length=40)


class ChatResponse(BaseModel):
    reply: str
    model: str
