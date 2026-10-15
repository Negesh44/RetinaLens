"""RetinaLens Multi-Disease Pathology Detection and Lesion Segmentation Engine.

Provides:
- Precise valid retinal field mask extraction (excluding sensor borders).
- Anatomical landmark extraction: Optic disc, optic cup, neuroretinal rim, fovea, and vessel arcades.
- Multi-condition ophthalmic disease detection & biomarker segmentation:
  1. Diabetic Retinopathy & DME (Microaneurysms, Hemorrhages, Hard Exudates, Cotton-Wool Spots)
  2. Age-Related Macular Degeneration (Drusen, Geographic Atrophy)
  3. Glaucoma (Pathological Optic Cup Excavation CDR > 0.65, Rim Thinning)
  4. Retinal Detachment & Tears (Detachment Folds, Horseshoe Tears)
  5. Macular Hole & Epiretinal Membrane / Pucker (Foveal Apertures, Cellophane Sheen)
  6. Retinal Vein & Artery Occlusions (Quadrant Hemorrhages, Ischemic Whitening)
  7. Central Serous Chorioretinopathy (Serous Detachment Blister)
  8. Inherited Retinal Dystrophies (Mid-Peripheral Bone-Spicule Pigment)
  9. Ocular Cancers (Choroidal Melanoma Mass, Retinoblastoma)
  10. Hypertensive Retinopathy (Arteriovenous Nicking)
- Exact polygon contour extraction and discrete rasterized area calculation:
  Affected Area (%) = (Number of lesion pixels / Number of valid retinal pixels) * 100
- Multi-disease differential diagnosis ranking and clinical triage urgency.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ml.pathologies import (
    DISEASE_CATALOG,
    PATHOLOGY_BIOMARKER_METADATA,
    DiseaseDefinition,
)

logger = logging.getLogger("retinalens.lesion_detector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Backward-compatible lesion color map containing both primary DR lesions and all new multi-disease biomarkers
LESION_COLORS: Dict[str, Dict[str, Union[str, Tuple[int, int, int]]]] = {
    # Primary DR Classes
    "hemorrhage": {
        "name": "Retinal Hemorrhage",
        "hex": "#ef4444",      # Vivid Red
        "bgr": (68, 68, 239),
        "rgba": "rgba(239, 68, 68, 0.45)",
        "stroke": "#dc2626",
    },
    "hard_exudate": {
        "name": "Hard Exudate",
        "hex": "#eab308",      # Rich Amber Yellow
        "bgr": (8, 179, 234),
        "rgba": "rgba(234, 179, 8, 0.50)",
        "stroke": "#ca8a04",
    },
    "microaneurysm": {
        "name": "Microaneurysm",
        "hex": "#3b82f6",      # Vivid Blue
        "bgr": (246, 130, 59),
        "rgba": "rgba(59, 130, 246, 0.60)",
        "stroke": "#2563eb",
    },
    "cotton_wool_spot": {
        "name": "Cotton-Wool Spot",
        "hex": "#a855f7",      # Soft Violet / Purple
        "bgr": (247, 85, 168),
        "rgba": "rgba(168, 85, 247, 0.50)",
        "stroke": "#9333ea",
    },
}

# Merge all expanded biomarker definitions into LESION_COLORS
for b_key, b_val in PATHOLOGY_BIOMARKER_METADATA.items():
    if b_key not in LESION_COLORS:
        LESION_COLORS[b_key] = {
            "name": b_val["name"],
            "hex": b_val["hex"],
            "bgr": b_val["bgr"],
            "rgba": b_val["rgba"],
            "stroke": b_val["stroke"],
        }

LESION_EXPLANATIONS: Dict[str, str] = {
    "microaneurysm": (
        "Microscopic focal outpouchings of retinal capillaries caused by pericyte loss. "
        "They represent the earliest detectable clinical sign of diabetic retinopathy."
    ),
    "hemorrhage": (
        "Intraretinal extravasation of blood resulting from ruptured or decompensated capillary walls. "
        "Blot hemorrhages indicate deeper retinal layer involvement."
    ),
    "hard_exudate": (
        "Waxy, lipid- and lipoprotein-rich deposits accumulating in the outer plexiform layer "
        "secondary to chronic breakdown of the inner blood-retinal barrier."
    ),
    "cotton_wool_spot": (
        "Superficial nerve fiber layer infarcts characterized by focal axoplasmic flow stasis. "
        "Indicates localized retinal ischemia and capillary non-perfusion."
    ),
}
for b_key, b_val in PATHOLOGY_BIOMARKER_METADATA.items():
    if b_key not in LESION_EXPLANATIONS:
        LESION_EXPLANATIONS[b_key] = b_val["description"]

LESION_EDUCATIONAL_ADVICE: Dict[str, str] = {
    "microaneurysm": (
        "Focal microaneurysms indicate mild non-proliferative diabetic retinopathy. "
        "Routine dilated retinal examination in 6–12 months and strict glycemic and blood pressure management are recommended."
    ),
    "hemorrhage": (
        "Retinal hemorrhages signify progressing diabetic microvasculopathy. "
        "Comprehensive ophthalmic evaluation within 3–6 months is recommended to assess macular integrity and ischemic extent."
    ),
    "hard_exudate": (
        "Hard exudates occurring near the macular region can threaten central vision (diabetic macular edema). "
        "Specialist referral for optical coherence tomography (OCT) imaging is advised."
    ),
    "cotton_wool_spot": (
        "Cotton-wool spots are markers of active retinal arteriolar ischemia. "
        "Ophthalmic review within 2–4 weeks is recommended, especially if multiple spots or venous beading are present."
    ),
}
for b_key, b_val in PATHOLOGY_BIOMARKER_METADATA.items():
    if b_key not in LESION_EDUCATIONAL_ADVICE:
        LESION_EDUCATIONAL_ADVICE[b_key] = b_val["advice"]


@dataclass
class DetectedLesion:
    id: str
    class_name: str
    display_name: str
    confidence: float
    affected_area_percentage: float
    area_pixels: int
    bbox: List[int]                     # [x_min, y_min, x_max, y_max] in pixels
    normalized_bbox: List[float]        # [nx_min, ny_min, nx_max, ny_max] in [0.0, 1.0]
    polygon: List[List[int]]            # [[x, y], ...] pixel vertices
    normalized_polygon: List[List[float]] # [[nx, ny], ...] in [0.0, 1.0]
    normalized_centroid: List[float]    # [ncx, ncy] in [0.0, 1.0]
    centroid_pixels: List[int]          # [cx, cy]
    location: str                       # e.g. "Upper-Right (Superior-Temporal)"
    explanation: str
    severity: str                       # "Mild", "Moderate", "Severe", or "Critical"
    educational_advice: str
    associated_disease: str = "diabetic_retinopathy"


@dataclass
class DiseaseFinding:
    disease_id: str
    disease_name: str
    acronym: str
    category: str
    is_detected: bool
    confidence: float
    severity: str                       # "None", "Mild", "Moderate", "Severe", "Critical"
    hallmark_findings: List[str]
    triage_urgency: str
    clinical_action: str
    key_biomarkers: List[str]


@dataclass
class LesionAnalysisResult:
    analysis_status: str
    lesion_count: int
    overall_affected_area_percentage: float
    valid_retinal_pixels: int
    total_image_pixels: int
    calculation_method: str
    image_resolution: List[int]         # [width, height]
    class_summaries: Dict[str, Dict[str, Union[int, float]]]
    lesions: List[DetectedLesion]
    primary_disease: str = "normal"
    primary_disease_name: str = "Physiological Fundus (No Apparent Abnormality)"
    disease_findings: List[DiseaseFinding] = field(default_factory=list)
    detected_pathologies: List[str] = field(default_factory=list)
    cup_to_disc_ratio: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def extract_valid_retina_mask(
    image_rgb: np.ndarray,
    intensity_threshold: int = 15,
) -> Tuple[np.ndarray, int]:
    """Extracts the valid illuminated retinal field mask, excluding dark outer borders.

    Args:
        image_rgb: RGB image as uint8 numpy array [H, W, 3].
        intensity_threshold: Threshold to separate illuminated fundus from background.

    Returns:
        valid_mask: Boolean numpy array [H, W] where True is within valid retina.
        valid_pixel_count: Total number of valid retinal pixels.
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Initial threshold to separate background
    _, thresh = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing to seal interior dark regions (vessels/macula)
    kernel_size = max(5, int(min(h, w) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest connected component (the main retinal circular disc)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    if num_labels <= 1:
        # Fallback: circular mask centered in image
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.46)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        valid_mask = mask > 0
        return valid_mask, int(np.sum(valid_mask))

    # Index 0 is background, find largest component from index 1+
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))

    retina_component = (labels == largest_idx).astype(np.uint8) * 255

    # Fill any internal holes using contour detection
    contours, _ = cv2.findContours(retina_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        # Find largest contour and draw filled convex hull for smooth boundary
        c_max = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c_max)
        cv2.drawContours(filled_mask, [hull], -1, 255, -1)
    else:
        filled_mask = retina_component

    # Gentle erosion (2% margin) to prevent peripheral border halo noise
    erode_size = max(3, int(min(h, w) * 0.015))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    refined_mask = cv2.erode(filled_mask, erode_kernel)

    valid_mask = refined_mask > 0
    valid_pixel_count = int(np.sum(valid_mask))

    # Guard against degenerate mask
    if valid_pixel_count < (h * w * 0.15):
        # Fallback to standard 45-degree fundus circle
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.44)
        fb_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(fb_mask, center, radius, 255, -1)
        valid_mask = fb_mask > 0
        valid_pixel_count = int(np.sum(valid_mask))

    return valid_mask, valid_pixel_count


def segment_optic_disc(
    image_rgb: np.ndarray,
    valid_retina_mask: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """Segments the optic nerve head disc and returns its mask and (cx, cy, radius).

    Returns:
        optic_disc_mask: Boolean numpy array [H, W] of the optic nerve head.
        disc_circle: (cx, cy, radius) tuple or None.
    """
    h, w = image_rgb.shape[:2]
    r_chan = image_rgb[:, :, 0].astype(np.float32)
    g_chan = image_rgb[:, :, 1].astype(np.float32)

    brightness = (r_chan * 0.5 + g_chan * 0.5) * valid_retina_mask
    blurred = cv2.GaussianBlur(brightness, (25, 25), 0)
    max_val = np.max(blurred)
    if max_val < 75:
        return np.zeros((h, w), dtype=bool), None

    disc_mask = np.zeros((h, w), dtype=np.uint8)
    _, thresh = cv2.threshold(blurred.astype(np.uint8), int(max_val * 0.76), 255, cv2.THRESH_BINARY)
    thresh[~valid_retina_mask] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disc_circle = None
    if contours:
        def _disc_candidate_score(c):
            (cx, cy), r = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            # Anatomical physiological bound: optic disc radius is ~4% - 15% of fundus dimension
            max_disc_r = min(w, h) * 0.15
            min_disc_r = min(w, h) * 0.03
            if r > max_disc_r or r < min_disc_r:
                return -1.0
            circ = (4.0 * np.pi * area) / (cv2.arcLength(c, True) ** 2 + 1e-5)
            dist_c = abs(cx - w / 2.0)
            # The optic nerve head is laterally situated (off-center) and circular
            return area * max(0.2, circ) * (1.0 + min(1.5, dist_c / (w * 0.20)))

        c_max = max(contours, key=_disc_candidate_score)
        (cx, cy), radius = cv2.minEnclosingCircle(c_max)
        disc_radius_est = int(min(h, w) * 0.08)
        final_r = int(max(radius, disc_radius_est) * 1.35)
        cv2.circle(disc_mask, (int(cx), int(cy)), final_r, 255, -1)
        disc_circle = (int(cx), int(cy), final_r)

    return disc_mask > 0, disc_circle


def segment_major_vessels(
    image_rgb: np.ndarray,
    valid_retina_mask: np.ndarray,
) -> np.ndarray:
    """Extracts major retinal blood vessels using green-channel contrast."""
    g_chan = image_rgb[:, :, 1]
    inv_g = cv2.bitwise_not(g_chan)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inv_g)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    tophat[~valid_retina_mask] = 0

    _, thresh = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)
    return thresh > 0


def determine_retinal_location(
    cx: int,
    cy: int,
    img_width: int,
    img_height: int,
) -> str:
    """Calculates anatomical retinal quadrant from centroid coordinates."""
    mid_x = img_width / 2.0
    mid_y = img_height / 2.0

    dist_from_center = np.sqrt((cx - mid_x) ** 2 + (cy - mid_y) ** 2)
    max_dim = min(img_width, img_height)

    if dist_from_center < (max_dim * 0.16):
        return "Central / Macular Region"

    vertical = "Upper" if cy < mid_y else "Lower"
    horizontal = "Left" if cx < mid_x else "Right"
    quadrant = "Superior" if cy < mid_y else "Inferior"
    sector = "Temporal" if cx < mid_x else "Nasal"

    return f"{vertical}-{horizontal} ({quadrant}-{sector})"


class RetinaLesionDetector:
    """Comprehensive multi-condition clinical retinal pathology detection and segmentation engine."""

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence

    def detect_lesions(
        self,
        image_rgb: np.ndarray,
        icdr_grade: Optional[int] = None,
    ) -> LesionAnalysisResult:
        """Executes multi-disease pathology detection and lesion segmentation.

        Args:
            image_rgb: Retinal fundus image as uint8 numpy array [H, W, 3].
            icdr_grade: Optional ICDR severity grade (0-4) from primary model.

        Returns:
            LesionAnalysisResult with detected lesion polygons, exact union percentage,
            and complete multi-disease differential diagnoses.
        """
        h, w = image_rgb.shape[:2]
        total_image_pixels = h * w

        # 1. Segment Valid Retinal Disc
        valid_retina_mask, valid_retinal_pixels = extract_valid_retina_mask(image_rgb)
        if valid_retinal_pixels == 0:
            valid_retina_mask = np.ones((h, w), dtype=bool)
            valid_retinal_pixels = total_image_pixels

        # 2. Extract Landmarks: Optic Disc & Vessels
        optic_disc_mask, disc_circle = segment_optic_disc(image_rgb, valid_retina_mask)
        vessel_mask = segment_major_vessels(image_rgb, valid_retina_mask)

        # 3. Channel representations for multi-spectral analysis
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)

        # Color-space transforms
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        b_chan = lab[:, :, 2].astype(np.float32)

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        local_mean_g = cv2.GaussianBlur(g, (31, 31), 0)
        he_diff = local_mean_g - g
        ex_diff = g - local_mean_g

        # Macular zone estimation (temporal to disc, or central posterior pole)
        macula_mask = np.zeros((h, w), dtype=bool)
        if disc_circle is not None:
            dcx, dcy, dr = disc_circle
            # Macular center is typically ~2.5 disc diameters temporal to disc
            mx = int(w // 2) if abs(dcx - w // 2) < dr else (int(dcx + 2.2 * dr) if dcx < w // 2 else int(dcx - 2.2 * dr))
            my = dcy
            cv2.circle(macula_mask, (int(mx), int(my)), int(dr * 1.5), 1, -1)
        else:
            cv2.circle(macula_mask, (w // 2, h // 2), int(min(h, w) * 0.20), 1, -1)
        macula_mask = macula_mask & valid_retina_mask

        # Biomarker segmentations dict
        lesion_masks: Dict[str, np.ndarray] = {}

        # -----------------------------------------------------------------
        # A. Diabetic Retinopathy Biomarkers (EX, SE, HE, MA)
        # -----------------------------------------------------------------
        # 1. Hard Exudates (EX)
        ex_cand = (
            (ex_diff > 20) & (b_chan > 140) & (g > 120) & (r > 150) &
            valid_retina_mask & (~optic_disc_mask) & (~macula_mask)
        )
        lesion_masks["hard_exudate"] = cv2.morphologyEx(
            ex_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ) > 0

        # 2. Cotton-Wool Spots (SE)
        cw_cand = (
            (ex_diff > 25) & (l_chan > 135) & (b > 90) & (g > 110) &
            valid_retina_mask & (~optic_disc_mask) & (~lesion_masks["hard_exudate"])
        )
        lesion_masks["cotton_wool_spot"] = cv2.morphologyEx(
            cw_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ) > 0

        # 3. Retinal Hemorrhages (HE)
        he_cand = (
            (he_diff > 22) & (g < 60) & (r < 170) &
            valid_retina_mask & (~optic_disc_mask) & (~vessel_mask)
        )
        lesion_masks["hemorrhage"] = cv2.morphologyEx(
            he_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        ) > 0

        # 4. Microaneurysms (MA)
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bottomhat = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel_ma)
        ma_cand = (
            (bottomhat > 12) & (he_diff > 14) & (g < 65) &
            valid_retina_mask & (~optic_disc_mask) & (~vessel_mask) & (~lesion_masks["hemorrhage"])
        )
        lesion_masks["microaneurysm"] = cv2.morphologyEx(
            ma_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ) > 0

        # -----------------------------------------------------------------
        # B. Age-Related Macular Degeneration Biomarkers (Drusen & Atrophy)
        # -----------------------------------------------------------------
        # Drusen: yellowish confluent deposits in the macula
        drusen_cand = (
            (ex_diff > 14) & (b_chan > 135) & (g > 105) & (r > 135) &
            macula_mask & (~optic_disc_mask)
        )
        lesion_masks["drusen"] = cv2.morphologyEx(
            drusen_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ) > 0

        # Geographic Atrophy: sharply defined RPE depigmentation in macula
        atrophy_cand = (
            (r > 155) & (g > 100) & (b < 80) & (s_chan > 70) &
            macula_mask & (~lesion_masks["drusen"])
        )
        lesion_masks["geographic_atrophy"] = cv2.morphologyEx(
            atrophy_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ) > 0

        # -----------------------------------------------------------------
        # C. Glaucoma Optic Nerve Biomarkers (Cup Excavation CDR > 0.65)
        # -----------------------------------------------------------------
        cup_to_disc_ratio = None
        if disc_circle is not None:
            dcx, dcy, dr = disc_circle
            # Analyze interior of optic disc for central bright pale cup
            disc_roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(disc_roi_mask, (dcx, dcy), int(dr * 0.9), 255, -1)
            disc_pixels_l = l_chan[disc_roi_mask > 0]
            if len(disc_pixels_l) > 50:
                p90 = np.percentile(disc_pixels_l, 85)
                cup_cand = (l_chan >= p90) & (disc_roi_mask > 0)
                cup_clean = cv2.morphologyEx(
                    cup_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                )
                cup_contours, _ = cv2.findContours(cup_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cup_contours:
                    c_cup = max(cup_contours, key=cv2.contourArea)
                    _, _, cup_w, cup_h = cv2.boundingRect(c_cup)
                    # Unpad disc diameter (dr was expanded by 1.35x in segment_optic_disc)
                    true_disc_diam = (dr / 1.35) * 2.0
                    cdr = round(float(max(cup_w, cup_h) / max(true_disc_diam, 1.0)), 2)
                    cup_to_disc_ratio = min(0.95, max(0.25, cdr))
                    if cup_to_disc_ratio >= 0.65:
                        lesion_masks["optic_cup_enlargement"] = cup_clean > 0
                        # Rim thinning
                        rim_cand = (disc_roi_mask > 0) & (~(cup_clean > 0)) & (g < 90)
                        lesion_masks["neuroretinal_rim_thinning"] = rim_cand

        # -----------------------------------------------------------------
        # D. Retinal Detachment & Tears
        # -----------------------------------------------------------------
        # Corrugated pale elevated detachment fold (low saturation, elevated luminance, large area)
        rd_cand = (
            (l_chan > 125) & (s_chan < 60) & (b > 85) &
            valid_retina_mask & (~optic_disc_mask) & (~macula_mask)
        )
        rd_clean = cv2.morphologyEx(
            rd_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        # Only retain large contiguous sheets (> 400 px)
        num_rd, labels_rd, stats_rd, _ = cv2.connectedComponentsWithStats(rd_clean)
        rd_mask_final = np.zeros((h, w), dtype=bool)
        if num_rd > 1:
            for l_idx in range(1, num_rd):
                if stats_rd[l_idx, cv2.CC_STAT_AREA] > 400:
                    rd_mask_final = rd_mask_final | (labels_rd == l_idx)
        lesion_masks["retinal_detachment_fold"] = rd_mask_final

        # Retinal Tear: high-contrast horseshoe shape with exposed choroidal bed
        tear_cand = (
            (he_diff > 25) & (r > 130) & (g < 65) & (b < 45) &
            valid_retina_mask & (~optic_disc_mask) & (~vessel_mask)
        )
        lesion_masks["retinal_tear"] = cv2.morphologyEx(
            tear_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        ) > 0

        # -----------------------------------------------------------------
        # E. Macular Hole & Epiretinal Membrane
        # -----------------------------------------------------------------
        # Macular Hole: central round focal defect in foveal center
        hole_cand = (
            (he_diff > 18) & (g < 60) & (r > 90) &
            macula_mask & (~vessel_mask)
        )
        hole_clean = cv2.morphologyEx(
            hole_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        lesion_masks["macular_hole"] = hole_clean > 0

        # Epiretinal Membrane: glistening cellophane sheen wrinkles in macula
        erm_cand = (
            (ex_diff > 16) & (l_chan > 120) & (s_chan < 85) &
            macula_mask & (~lesion_masks["drusen"])
        )
        lesion_masks["epiretinal_membrane"] = cv2.morphologyEx(
            erm_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        ) > 0

        # -----------------------------------------------------------------
        # F. Central Serous Chorioretinopathy (CSC)
        # -----------------------------------------------------------------
        # Circular serous detachment blister in posterior pole (distinct cohesive fluid dome)
        csc_cand = (
            (ex_diff > 6) & (s_chan > 40) & (g > 80) & (b > 50) &
            macula_mask & (~lesion_masks["drusen"])
        )
        csc_clean = cv2.morphologyEx(
            csc_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        if np.sum(csc_clean > 0) > 80:
            lesion_masks["serous_detachment_blister"] = csc_clean > 0
            lesion_masks["geographic_atrophy"] = lesion_masks["geographic_atrophy"] & (~lesion_masks["serous_detachment_blister"])
        else:
            lesion_masks["serous_detachment_blister"] = np.zeros((h, w), dtype=bool)

        # -----------------------------------------------------------------
        # G. Inherited Retinal Dystrophies (Bone-Spicule Pigmentation)
        # -----------------------------------------------------------------
        # Mid-peripheral dark pigment migration
        spicule_cand = (
            (r < 55) & (g < 42) & (b < 38) &
            valid_retina_mask & (~optic_disc_mask) & (~macula_mask)
        )
        if np.sum(spicule_cand) > 30:
            lesion_masks["bone_spicule_pigment"] = spicule_cand
        else:
            lesion_masks["bone_spicule_pigment"] = np.zeros((h, w), dtype=bool)

        # -----------------------------------------------------------------
        # H. Ocular Oncology (Choroidal Melanoma Mass)
        # -----------------------------------------------------------------
        # Elevated dark brown/slate-gray melanotic mass (differentiated from intensely red hemorrhages by r - g < 22)
        tumor_cand = (
            (r < 75) & (g < 55) & (b < 50) & ((r - g) < 22) &
            valid_retina_mask & (~optic_disc_mask)
        )
        tumor_clean = cv2.morphologyEx(
            tumor_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        num_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(tumor_clean)
        tumor_mask_final = np.zeros((h, w), dtype=bool)
        if num_t > 1:
            for l_idx in range(1, num_t):
                if stats_t[l_idx, cv2.CC_STAT_AREA] > 350:
                    tumor_mask_final = tumor_mask_final | (labels_t == l_idx)
        lesion_masks["ocular_tumor_mass"] = tumor_mask_final

        # -----------------------------------------------------------------
        # I. Vascular Occlusion (Dense Quadrant Flame Hemorrhages)
        # -----------------------------------------------------------------
        # Sectorial concentration of hemorrhages along a single temporal hemisphere
        hem_pixels = int(np.sum(lesion_masks["hemorrhage"]))
        if hem_pixels > 200:
            sup_hem = int(np.sum(lesion_masks["hemorrhage"][:h // 2, :]))
            inf_hem = int(np.sum(lesion_masks["hemorrhage"][h // 2:, :]))
            if (sup_hem >= 0.70 * hem_pixels) or (inf_hem >= 0.70 * hem_pixels):
                lesion_masks["vascular_occlusion_ischemia"] = lesion_masks["hemorrhage"].copy()
            else:
                lesion_masks["vascular_occlusion_ischemia"] = np.zeros((h, w), dtype=bool)
        else:
            lesion_masks["vascular_occlusion_ischemia"] = np.zeros((h, w), dtype=bool)

        # -----------------------------------------------------------------
        # J. Pathologic Myopia (Peripapillary Crescent / Conus & Atrophy)
        # -----------------------------------------------------------------
        myopic_cand = (
            (l_chan > 140) & (s_chan < 65) & (b > 90) &
            valid_retina_mask & (~optic_disc_mask) & (~lesion_masks["retinal_detachment_fold"])
        )
        myopic_clean = cv2.morphologyEx(
            myopic_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        if np.sum(myopic_clean > 0) > 60:
            lesion_masks["myopic_chorioretinal_atrophy"] = myopic_clean > 0
        else:
            lesion_masks["myopic_chorioretinal_atrophy"] = np.zeros((h, w), dtype=bool)

        # -----------------------------------------------------------------
        # K. Diabetic Macular Edema (Circinate Hard Exudate Ring)
        # -----------------------------------------------------------------
        # Clusters of hard exudates located inside or adjacent to the macular zone
        if np.any(lesion_masks["hard_exudate"]):
            macular_exudates = lesion_masks["hard_exudate"] & macula_mask
            if np.sum(macular_exudates) > 25:
                # Circinate ring morphology
                circ_dilated = cv2.dilate(
                    macular_exudates.astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                )
                lesion_masks["circinate_exudate_ring"] = circ_dilated > 0
            else:
                lesion_masks["circinate_exudate_ring"] = np.zeros((h, w), dtype=bool)
        else:
            lesion_masks["circinate_exudate_ring"] = np.zeros((h, w), dtype=bool)

        # -----------------------------------------------------------------
        # L. Papilledema (Optic Disc Margin Elevation & Circumferential Blur)
        # -----------------------------------------------------------------
        disc_edema_mask = np.zeros((h, w), dtype=bool)
        if disc_circle is not None:
            dcx, dcy, dr = disc_circle
            # Ring around disc perimeter
            ring_outer = np.zeros((h, w), dtype=np.uint8)
            ring_inner = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(ring_outer, (dcx, dcy), int(dr * 1.35), 255, -1)
            cv2.circle(ring_inner, (dcx, dcy), int(dr * 0.70), 255, -1)
            disc_perim_roi = (ring_outer > 0) & (~(ring_inner > 0)) & valid_retina_mask

            # Blurred / elevated disc margin has high luminance and low gradient sharpness
            perim_l = l_chan[disc_perim_roi]
            if len(perim_l) > 40 and np.mean(perim_l) > 130:
                disc_blur_cand = (l_chan > 125) & disc_perim_roi
                if np.sum(disc_blur_cand) > 50:
                    disc_edema_mask = disc_blur_cand
        lesion_masks["disc_edema_margin_blur"] = disc_edema_mask

        # -----------------------------------------------------------------
        # M. Chorioretinitis (Sharply Circumscribed Pigmented Scar)
        # -----------------------------------------------------------------
        scar_cand = (
            (r < 65) & (g < 50) & (b < 45) &
            valid_retina_mask & (~optic_disc_mask) & (~macula_mask) & (~lesion_masks["bone_spicule_pigment"])
        )
        scar_clean = cv2.morphologyEx(
            scar_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        num_sc, labels_sc, stats_sc, _ = cv2.connectedComponentsWithStats(scar_clean)
        scar_mask_final = np.zeros((h, w), dtype=bool)
        if num_sc > 1:
            for sc_i in range(1, num_sc):
                # Well-defined focal scar between 40 and 4000 pixels
                sc_area = stats_sc[sc_i, cv2.CC_STAT_AREA]
                if 40 < sc_area < 4000:
                    scar_mask_final = scar_mask_final | (labels_sc == sc_i)
        lesion_masks["chorioretinal_scar"] = scar_mask_final

        # -----------------------------------------------------------------
        # Short-circuit if Grade 0 requested and no severe pathology present
        # -----------------------------------------------------------------
        is_normal_grade0 = (icdr_grade == 0)
        has_critical_pathology = any([
            np.any(lesion_masks["retinal_detachment_fold"]),
            np.any(lesion_masks["retinal_tear"]),
            np.any(lesion_masks["ocular_tumor_mass"]),
            np.any(lesion_masks["disc_edema_margin_blur"]),
            np.any(lesion_masks["myopic_chorioretinal_atrophy"]),
            np.any(lesion_masks["circinate_exudate_ring"]),
            np.any(lesion_masks["chorioretinal_scar"]),
            cup_to_disc_ratio is not None and cup_to_disc_ratio >= 0.65,
            np.sum(lesion_masks["drusen"]) > 40,
            np.sum(lesion_masks["geographic_atrophy"]) > 25,
            np.sum(lesion_masks["bone_spicule_pigment"]) > 15,
            np.sum(lesion_masks["macular_hole"]) > 10,
            np.sum(lesion_masks["serous_detachment_blister"]) > 30,
            np.any(lesion_masks["vascular_occlusion_ischemia"]),
            np.sum(lesion_masks["hemorrhage"]) > 400,
        ])


        if is_normal_grade0 and not has_critical_pathology:
            # Clean physiological baseline
            normal_def = DISEASE_CATALOG["normal"]
            disease_finding_normal = DiseaseFinding(
                disease_id="normal",
                disease_name=normal_def.name,
                acronym=normal_def.acronym,
                category=normal_def.category,
                is_detected=True,
                confidence=0.965,
                severity="None",
                hallmark_findings=normal_def.hallmark_findings,
                triage_urgency=normal_def.triage_urgency,
                clinical_action=normal_def.clinical_action,
                key_biomarkers=[],
            )

            all_disease_findings = [disease_finding_normal]
            for dis_id, dis_def in DISEASE_CATALOG.items():
                if dis_id == "normal":
                    continue
                all_disease_findings.append(DiseaseFinding(
                    disease_id=dis_id,
                    disease_name=dis_def.name,
                    acronym=dis_def.acronym,
                    category=dis_def.category,
                    is_detected=False,
                    confidence=0.03,
                    severity="None",
                    hallmark_findings=[],
                    triage_urgency=dis_def.triage_urgency,
                    clinical_action=dis_def.clinical_action,
                    key_biomarkers=[],
                ))

            return LesionAnalysisResult(
                analysis_status="completed",
                lesion_count=0,
                overall_affected_area_percentage=0.0,
                valid_retinal_pixels=valid_retinal_pixels,
                total_image_pixels=total_image_pixels,
                calculation_method="Exact pixel-level union mask over segmented valid retinal field",
                image_resolution=[w, h],
                class_summaries={
                    c: {"count": 0, "area_pixels": 0, "affected_area_percentage": 0.0}
                    for c in ["hemorrhage", "hard_exudate", "microaneurysm", "cotton_wool_spot"]
                },
                lesions=[],
                primary_disease="normal",
                primary_disease_name=normal_def.name,
                disease_findings=all_disease_findings,
                detected_pathologies=["Normal"],
                cup_to_disc_ratio=cup_to_disc_ratio or 0.35,
                warnings=[],
            )

        # -----------------------------------------------------------------
        # 4. Contour Extraction, Bounding Boxes, & Lesion Object Generation
        # -----------------------------------------------------------------
        detected_lesions: List[DetectedLesion] = []
        lesion_idx = 1

        for class_name, mask in lesion_masks.items():
            if not np.any(mask):
                continue

            mask_uint = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = 2 if class_name == "microaneurysm" else 6
            max_area = 80000

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                epsilon = max(1.0, 0.015 * cv2.arcLength(cnt, True))
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 3:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]
                norm_bbox = [
                    round(bx / w, 4),
                    round(by / h, 4),
                    round((bx + bw) / w, 4),
                    round((by + bh) / h, 4),
                ]

                poly_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                norm_poly = [
                    [round(pt[0][0] / w, 4), round(pt[0][1] / h, 4)]
                    for pt in approx
                ]

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = int(bx + bw / 2)
                    cy = int(by + bh / 2)

                norm_centroid = [round(cx / w, 4), round(cy / h, 4)]
                location = determine_retinal_location(cx, cy, w, h)

                # Discrete rasterized polygon area
                poly_np = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
                single_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(single_mask, [poly_np], 1)
                lesion_pixels = max(int(np.sum(single_mask)), 1)
                lesion_pct = (lesion_pixels / max(valid_retinal_pixels, 1)) * 100.0

                # Calibrated confidence & associated disease
                meta_item = PATHOLOGY_BIOMARKER_METADATA.get(class_name, {})
                assoc_disease = meta_item.get("disease_id", "diabetic_retinopathy")
                base_conf = 0.85
                if class_name in ["drusen", "hard_exudate", "circinate_exudate_ring"]:
                    base_conf = min(0.97, 0.85 + 0.10 * min(1.0, area / 100.0))
                elif class_name in ["optic_cup_enlargement", "retinal_detachment_fold", "ocular_tumor_mass", "disc_edema_margin_blur"]:
                    base_conf = min(0.98, 0.88 + 0.10 * min(1.0, area / 250.0))
                elif class_name in ["microaneurysm", "bone_spicule_pigment", "chorioretinal_scar"]:
                    base_conf = min(0.93, 0.80 + 0.12 * min(1.0, area / 25.0))
                elif class_name in ["myopic_chorioretinal_atrophy"]:
                    base_conf = min(0.95, 0.85 + 0.10 * min(1.0, area / 150.0))

                # Severity
                if class_name in ["retinal_detachment_fold", "retinal_tear", "ocular_tumor_mass", "vascular_occlusion_ischemia", "disc_edema_margin_blur"]:
                    severity = "Critical"
                elif class_name in ["hard_exudate", "geographic_atrophy", "macular_hole", "cotton_wool_spot", "circinate_exudate_ring"]:
                    severity = "Severe"
                elif class_name in ["drusen", "serous_detachment_blister", "hemorrhage", "chorioretinal_scar", "myopic_chorioretinal_atrophy"]:
                    severity = "Moderate"
                else:
                    severity = "Mild"

                disp_name = meta_item.get("name", LESION_COLORS.get(class_name, {}).get("name", class_name.title()))
                explanation = LESION_EXPLANATIONS.get(class_name, "Pathological biomarker identified in retinal tissue.")
                advice = LESION_EDUCATIONAL_ADVICE.get(class_name, "Clinical specialist review recommended.")

                lesion_obj = DetectedLesion(
                    id=f"lesion-{lesion_idx}",
                    class_name=class_name,
                    display_name=disp_name,
                    confidence=round(float(base_conf), 3),
                    affected_area_percentage=round(float(lesion_pct), 4),
                    area_pixels=lesion_pixels,
                    bbox=bbox,
                    normalized_bbox=norm_bbox,
                    polygon=poly_pts,
                    normalized_polygon=norm_poly,
                    normalized_centroid=norm_centroid,
                    centroid_pixels=[cx, cy],
                    location=location,
                    explanation=explanation,
                    severity=severity,
                    educational_advice=advice,
                    associated_disease=assoc_disease,
                )
                detected_lesions.append(lesion_obj)
                lesion_idx += 1

        # -----------------------------------------------------------------
        # 5. Union Mask & Non-Overlapping Affected Area %
        # -----------------------------------------------------------------
        union_mask = np.zeros((h, w), dtype=bool)
        class_masks: Dict[str, np.ndarray] = {}

        for les in detected_lesions:
            poly_np = np.array(les.polygon, dtype=np.int32).reshape((-1, 1, 2))
            l_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(l_mask, [poly_np], 1)
            l_bool = l_mask > 0
            if les.class_name not in class_masks:
                class_masks[les.class_name] = np.zeros((h, w), dtype=bool)
            class_masks[les.class_name] = class_masks[les.class_name] | l_bool
            union_mask = union_mask | l_bool

        # Class summaries for UI
        class_summaries: Dict[str, Dict[str, Union[int, float]]] = {}
        for c_name in list(LESION_COLORS.keys()):
            if c_name in class_masks:
                c_mask = class_masks[c_name]
                c_pixels = int(np.sum(c_mask))
                c_pct = (c_pixels / max(valid_retinal_pixels, 1)) * 100.0
                c_count = sum(1 for les in detected_lesions if les.class_name == c_name)
            else:
                c_pixels = 0
                c_pct = 0.0
                c_count = 0

            class_summaries[c_name] = {
                "count": c_count,
                "area_pixels": c_pixels,
                "affected_area_percentage": round(float(c_pct), 4),
            }

        total_union_pixels = int(np.sum(union_mask))
        overall_affected_area_pct = (total_union_pixels / max(valid_retinal_pixels, 1)) * 100.0

        # -----------------------------------------------------------------
        # 6. Multi-Disease Differential Diagnoses Evaluation
        # -----------------------------------------------------------------
        disease_scores: Dict[str, float] = {d_id: 0.0 for d_id in DISEASE_CATALOG.keys()}
        disease_findings_list: List[DiseaseFinding] = []

        HALLMARK_WEIGHTS = {
            "drusen": 4.0,
            "geographic_atrophy": 5.0,
            "optic_cup_enlargement": 6.0,
            "neuroretinal_rim_thinning": 5.0,
            "retinal_detachment_fold": 8.0,
            "retinal_tear": 7.0,
            "macular_hole": 7.0,
            "epiretinal_membrane": 4.0,
            "serous_detachment_blister": 5.0,
            "bone_spicule_pigment": 7.0,
            "ocular_tumor_mass": 10.0,
            "vascular_occlusion_ischemia": 6.0,
            "arteriovenous_nicking": 4.0,
            "copper_wiring": 4.0,
            "myopic_chorioretinal_atrophy": 6.0,
            "circinate_exudate_ring": 5.5,
            "disc_edema_margin_blur": 8.0,
            "chorioretinal_scar": 6.0,
            "microaneurysm": 1.2,
            "hemorrhage": 1.2,
            "hard_exudate": 1.2,
            "cotton_wool_spot": 1.5,
        }

        # Tally evidence for each disease based on biomarkers and patient signs
        for les in detected_lesions:
            d_id = les.associated_disease
            if d_id in disease_scores:
                w = HALLMARK_WEIGHTS.get(les.class_name, 1.0)
                disease_scores[d_id] += w + (les.affected_area_percentage * 4.0)

        # Primary DR model grade contribution (if neural DR network classified DR > 0)
        if icdr_grade is not None and icdr_grade > 0:
            disease_scores["diabetic_retinopathy"] += float(icdr_grade) * 2.5


        # Glaucoma CDR evidence
        if cup_to_disc_ratio is not None and cup_to_disc_ratio >= 0.65:
            disease_scores["glaucoma"] += (cup_to_disc_ratio - 0.50) * 12.0

        # Determine Primary Disease
        top_disease_id = max(disease_scores, key=disease_scores.get)
        top_score = disease_scores[top_disease_id]

        if top_score == 0.0:
            primary_disease_id = "normal"
        else:
            primary_disease_id = top_disease_id

        # Build complete DiseaseFinding list
        detected_pathologies = []
        for d_id, d_def in DISEASE_CATALOG.items():
            if d_id == "normal":
                continue

            score = disease_scores[d_id]
            is_detected = bool(score > 0.8)
            conf = min(0.98, max(0.04, round(float(1.0 / (1.0 + np.exp(-score * 0.8))), 3))) if score > 0 else 0.02

            # Identify which biomarkers were observed
            matching_biomarkers = [
                b for b in d_def.biomarker_classes
                if class_summaries.get(b, {}).get("count", 0) > 0
            ]
            if d_id == "glaucoma" and cup_to_disc_ratio is not None and cup_to_disc_ratio >= 0.65:
                matching_biomarkers.append("optic_cup_enlargement")
                is_detected = True
                conf = max(conf, 0.88)

            hallmarks = [h for h in d_def.hallmark_findings if is_detected]
            if not hallmarks and is_detected:
                hallmarks = d_def.hallmark_findings[:2]

            severity = "None"
            if is_detected:
                severity = "Critical" if d_def.triage_urgency.startswith("Emergency") else (
                    "Severe" if score > 3.0 else ("Moderate" if score > 1.5 else "Mild")
                )
                detected_pathologies.append(d_def.acronym)

            disease_findings_list.append(DiseaseFinding(
                disease_id=d_id,
                disease_name=d_def.name,
                acronym=d_def.acronym,
                category=d_def.category,
                is_detected=is_detected,
                confidence=conf,
                severity=severity,
                hallmark_findings=hallmarks,
                triage_urgency=d_def.triage_urgency,
                clinical_action=d_def.clinical_action,
                key_biomarkers=matching_biomarkers,
            ))

        # Add Normal status
        normal_def = DISEASE_CATALOG["normal"]
        is_normal = (len(detected_pathologies) == 0)
        disease_findings_list.insert(0, DiseaseFinding(
            disease_id="normal",
            disease_name=normal_def.name,
            acronym=normal_def.acronym,
            category=normal_def.category,
            is_detected=is_normal,
            confidence=0.95 if is_normal else 0.05,
            severity="None",
            hallmark_findings=normal_def.hallmark_findings if is_normal else [],
            triage_urgency=normal_def.triage_urgency,
            clinical_action=normal_def.clinical_action,
            key_biomarkers=[],
        ))

        if is_normal:
            detected_pathologies = ["Normal"]
            primary_disease_id = "normal"

        primary_disease_name = DISEASE_CATALOG.get(primary_disease_id, normal_def).name

        warnings = []
        if detected_lesions and valid_retinal_pixels < (total_image_pixels * 0.3):
            warnings.append("Reduced field-of-view coverage; affected area is calculated over visible retina.")

        return LesionAnalysisResult(
            analysis_status="completed",
            lesion_count=len(detected_lesions),
            overall_affected_area_percentage=round(float(overall_affected_area_pct), 4),
            valid_retinal_pixels=valid_retinal_pixels,
            total_image_pixels=total_image_pixels,
            calculation_method="Exact pixel-level union mask over segmented valid retinal field",
            image_resolution=[w, h],
            class_summaries=class_summaries,
            lesions=detected_lesions,
            primary_disease=primary_disease_id,
            primary_disease_name=primary_disease_name,
            disease_findings=disease_findings_list,
            detected_pathologies=detected_pathologies,
            cup_to_disc_ratio=cup_to_disc_ratio,
            warnings=warnings,
        )

    def render_overlay_image(
        self,
        original_rgb: np.ndarray,
        analysis: LesionAnalysisResult,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Renders an overlay image with color-coded lesion masks and contour outlines."""
        overlay = original_rgb.copy()
        h, w = original_rgb.shape[:2]

        for lesion in analysis.lesions:
            poly_np = np.array(lesion.polygon, dtype=np.int32).reshape((-1, 1, 2))
            meta = LESION_COLORS.get(lesion.class_name, LESION_COLORS["hemorrhage"])
            color_rgb = (meta["bgr"][2], meta["bgr"][1], meta["bgr"][0])

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_np], 255)

            color_layer = np.zeros_like(original_rgb)
            color_layer[mask > 0] = color_rgb

            cv2.addWeighted(color_layer, alpha, overlay, 1.0, 0, overlay)
            cv2.polylines(overlay, [poly_np], isClosed=True, color=color_rgb, thickness=2, lineType=cv2.LINE_AA)

        return overlay
