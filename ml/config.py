"""Central Configuration Module for RetinaLens.

This module houses all system-wide constants, clinical grade mappings,
reproducibility seeds, and default threshold configurations.
"""

from pathlib import Path
from typing import Dict, List

# Single-source project naming constant (can be renamed here)
PROJECT_NAME: str = "RetinaLens"
PROJECT_VERSION: str = "1.0.0"

# Mandatory Clinical Disclaimer - Required across all screens and reports
CLINICAL_DISCLAIMER: str = (
    "RetinaLens is an investigational screening aid designed to assist qualified "
    "clinicians and trained healthcare personnel. It is NOT a medical device, does NOT "
    "provide an automated diagnosis, and does not replace comprehensive ophthalmic examination."
)

# Standardized International Clinical Diabetic Retinopathy (ICDR) 5-point scale
ICDR_GRADES: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

# Detailed clinical descriptions and recommended triage actions per grade
ICDR_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    0: {
        "title": "No Apparent Retinopathy",
        "action": "Routine annual rescreening recommended.",
        "findings": "No microaneurysms, hemorrhages, or exudates detected.",
        "referable": False,
    },
    1: {
        "title": "Mild Non-Proliferative DR",
        "action": "Follow-up screening in 6-12 months. Optimize glycemic & BP control.",
        "findings": "Microaneurysms only.",
        "referable": False,
    },
    2: {
        "title": "Moderate Non-Proliferative DR",
        "action": "Refer to ophthalmologist within 3-6 months.",
        "findings": "More than microaneurysms, but less than severe NPDR (blot hemorrhages, hard exudates, cotton wool spots).",
        "referable": True,
    },
    3: {
        "title": "Severe Non-Proliferative DR",
        "action": "Prompt ophthalmologist referral within 2-4 weeks.",
        "findings": "Any of: >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, prominent IRMA in 1+ quadrant.",
        "referable": True,
    },
    4: {
        "title": "Proliferative Diabetic Retinopathy",
        "action": "Urgent specialist referral within 24-48 hours (risk of acute vision loss).",
        "findings": "Neovascularization of the disc/retina, vitreous or preretinal hemorrhage.",
        "referable": True,
    },
}

# Grades >= 2 warrant clinical referral
REFERABLE_GRADE_THRESHOLD: int = 2

# Reproducibility Seed
RANDOM_SEED: int = 42

# Image Quality & Uncertainty Threshold Defaults
DEFAULT_QUALITY_THRESHOLD: float = 0.50
DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.60  # If max calibrated prob < 0.60, flag as uncertain

# Model Archetypes & Supported Backbones
SUPPORTED_BACKBONES: List[str] = [
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
]
DEFAULT_BACKBONE: str = "efficientnet_b0"

# Target Resolutions
BACKBONE_IMAGE_SIZES: Dict[str, int] = {
    "efficientnet_b0": 224,
    "efficientnet_b3": 380,
    "resnet50": 224,
}

# Paths Configuration
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CHARTS_DIR: Path = PROJECT_ROOT / "generated_charts"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
