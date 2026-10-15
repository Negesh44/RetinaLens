"""CPU inference for the trained retinal disease classifier (ONNX Runtime).

Two formats are supported:
* an ensemble description, ``checkpoints/retina_ensemble.json`` (written by ``ml.finalize_ensemble``):
  several models whose logits are averaged, per-condition Platt calibration, thresholds and a
  consistency rule, all tuned on validation data;
* a single model with its training report (``ml.train_multidisease``).

The held-out test metrics in the description are passed through so the UI can show how reliable
each prediction is.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ml.prepare_multidisease import crop_fundus

logger = logging.getLogger("retinalens.disease_classifier")

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
ENSEMBLE_REPORT = CHECKPOINTS / "retina_ensemble.json"
DEFAULT_MODEL = CHECKPOINTS / "retina_multidisease.onnx"
DEFAULT_REPORT = CHECKPOINTS / "retina_multidisease_report.json"
CAM_SIZE = 32


@dataclass
class ConditionPrediction:
    condition_id: str
    probability: float
    threshold: float
    detected: bool
    test_auc: Optional[float] = None
    test_sensitivity: Optional[float] = None
    test_specificity: Optional[float] = None
    test_positives: int = 0


@dataclass
class ClassifierResult:
    predictions: Dict[str, ConditionPrediction]
    abnormal_probability: float
    abnormal_detected: bool
    # Class activation maps in [0, 1], aligned to the cropped square fundus image, keyed by condition id.
    cams: Dict[str, np.ndarray] = field(default_factory=dict)
    # Maps the CAM's square frame back to original image coordinates: (x0, y0, side) in original pixels.
    crop_box: tuple = (0, 0, 0)
    model_name: str = ""


class RetinaDiseaseClassifier:
    def __init__(self, model_path: Optional[Path] = None, report_path: Optional[Path] = None):
        import onnxruntime as ort

        if model_path is None and report_path is None and ENSEMBLE_REPORT.exists():
            self.report = json.loads(ENSEMBLE_REPORT.read_text())
            members = [(CHECKPOINTS / m["onnx"], m["image_size"], m["backbone"]) for m in self.report["members"]]
        else:
            self.report = json.loads(Path(report_path or DEFAULT_REPORT).read_text())
            members = [(Path(model_path or DEFAULT_MODEL), self.report["image_size"], self.report["backbone"])]

        self.classes: List[str] = self.report["classes"]
        self.thresholds: Dict[str, float] = self.report["thresholds"]
        self.calibration: Dict[str, List[float]] = self.report.get("calibration", {})
        rule = self.report.get("consistency_rule", {})
        self.gate: float = rule.get("gate", 0.0)
        self.override: float = rule.get("override", 2.0)
        self.mean = np.array(self.report["mean"], np.float32)
        self.std = np.array(self.report["std"], np.float32)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.members = [
            (ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"]), size)
            for path, size, _ in members
        ]
        names = [f"{backbone.split('.')[0]} @ {size}px" for _, size, backbone in members]
        self.model_name = names[0] if len(names) == 1 else f"ensemble of {len(names)}: " + " + ".join(names)
        logger.info("Loaded disease classifier (%s) with %d classes", self.model_name, len(self.classes))

    @staticmethod
    def available() -> bool:
        return ENSEMBLE_REPORT.exists() or (DEFAULT_MODEL.exists() and DEFAULT_REPORT.exists())

    def _crop_box(self, rgb: np.ndarray):
        """Square box (x0, y0, side) in original pixels matching crop_fundus's crop and padding."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.morphologyEx((gray > 12).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs) > 0.05 * mask.size:
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        else:
            x1, y1, x2, y2 = 0, 0, rgb.shape[1], rgb.shape[0]
        w, h = x2 - x1, y2 - y1
        side = max(w, h)
        return int(x1 - (side - w) // 2), int(y1 - (side - h) // 2), int(side)

    def _run_member(self, session, size: int, square: np.ndarray):
        x = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        # Horizontal-flip test-time augmentation, as in evaluation.
        batch = np.ascontiguousarray(np.concatenate([x, x[..., ::-1]]), dtype=np.float32)
        logits, cams = session.run(["logits", "cams"], {"image": batch})
        cam = np.maximum((cams[0] + cams[1][..., ::-1]) / 2.0, 0)
        # Normalise each class map to [0, 1] and bring all members onto one grid before averaging.
        peak = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        cam = np.where(peak > 0, cam / np.maximum(peak, 1e-8), 0.0).astype(np.float32)
        cam = np.stack([cv2.resize(m, (CAM_SIZE, CAM_SIZE), interpolation=cv2.INTER_LINEAR) for m in cam])
        return logits.mean(axis=0), cam

    def predict(self, rgb: np.ndarray) -> ClassifierResult:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        square = cv2.cvtColor(crop_fundus(bgr), cv2.COLOR_BGR2RGB)
        outs = [self._run_member(session, size, square) for session, size in self.members]
        z = np.mean([o[0] for o in outs], axis=0)
        cam = np.mean([o[1] for o in outs], axis=0)

        probs = np.empty(len(self.classes))
        for j, c in enumerate(self.classes):
            a, b = self.calibration.get(c, (1.0, 0.0))
            probs[j] = 1.0 / (1.0 + np.exp(-(a * z[j] + b)))
        p_abnormal = float(probs[self.classes.index("abnormal")])

        test = self.report.get("test", {})
        preds: Dict[str, ConditionPrediction] = {}
        cam_maps: Dict[str, np.ndarray] = {}
        for j, c in enumerate(self.classes):
            t = test.get(c, {})
            p = float(probs[j])
            detected = p >= self.thresholds[c]
            if c != "abnormal":
                # Consistency rule: the overall abnormality output must agree unless the evidence is strong.
                detected = detected and (p_abnormal >= self.gate or p >= self.override)
            preds[c] = ConditionPrediction(
                condition_id=c,
                probability=p,
                threshold=float(self.thresholds[c]),
                detected=bool(detected),
                test_auc=t.get("auc"),
                test_sensitivity=t.get("sensitivity"),
                test_specificity=t.get("specificity"),
                test_positives=int(t.get("n_pos", 0)),
            )
            m = np.maximum(cam[j], 0)
            cam_maps[c] = (m / m.max()).astype(np.float32) if m.max() > 0 else m.astype(np.float32)

        abnormal = preds.pop("abnormal")
        return ClassifierResult(
            predictions=preds,
            abnormal_probability=abnormal.probability,
            abnormal_detected=abnormal.detected,
            cams=cam_maps,
            crop_box=self._crop_box(rgb),
            model_name=self.model_name,
        )


def _region_location(cx: float, cy: float, w: int, h: int) -> str:
    dx, dy = cx / w - 0.5, cy / h - 0.5
    if (dx * dx + dy * dy) ** 0.5 < 0.14:
        return "Central region"
    return f"{'Upper' if dy < 0 else 'Lower'}-{'left' if dx < 0 else 'right'} region"


def attention_regions(result: ClassifierResult, condition_id: str, name: str, image_shape, valid_pixels: int,
                      advice: str, max_regions: int = 3, level: float = 0.6) -> List[dict]:
    """Hot spots of one condition's activation map, as lesion-shaped dicts in original image coordinates."""
    h, w = image_shape[:2]
    x0, y0, side = result.crop_box
    if side <= 0:
        return []
    cam_sq = cv2.resize(result.cams[condition_id], (side, side), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((h, w), np.float32)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw, ch = min(side - sx0, w - dx0), min(side - sy0, h - dy0)
    if cw <= 0 or ch <= 0:
        return []
    full[dy0:dy0 + ch, dx0:dx0 + cw] = cam_sq[sy0:sy0 + ch, sx0:sx0 + cw]

    mask = (full >= level).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:max_regions]
    pred = result.predictions[condition_id]
    regions = []
    for k, i in enumerate(order):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.003 * h * w:
            continue
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.004 * (h + w), True).reshape(-1, 2)
        x, y, bw, bh = (int(v) for v in cv2.boundingRect(contour))
        cx, cy = (float(v) for v in centroids[i])
        peak = float(full[labels == i].max())
        regions.append({
            "id": f"region-{condition_id}-{k + 1}",
            "class_name": "attention_region",
            "display_name": f"{name}: AI attention area",
            "confidence": round(pred.probability, 4),
            "affected_area_percentage": round(100.0 * area / max(valid_pixels, 1), 3),
            "area_pixels": area,
            "bbox": [x, y, x + bw, y + bh],
            "normalized_bbox": [x / w, y / h, (x + bw) / w, (y + bh) / h],
            "polygon": contour.tolist(),
            "normalized_polygon": [[px / w, py / h] for px, py in contour.tolist()],
            "normalized_centroid": [cx / w, cy / h],
            "centroid_pixels": [int(cx), int(cy)],
            "location": _region_location(cx, cy, w, h),
            "explanation": (
                f"The model's {name} prediction relies {'most' if k == 0 else 'strongly'} on this area "
                f"(relative attention {peak:.0%}). This shows where the AI looked; it is not a measured lesion outline."
            ),
            "severity": "",
            "educational_advice": advice,
            "associated_disease": condition_id,
        })
    return regions


def apply_to_analysis(result: ClassifierResult, analysis, image_shape) -> None:
    """Replace the heuristic disease scoring in a LesionAnalysisResult with the trained model's output.

    Findings come from the classifier; the colour-heuristic lesion candidates are replaced by the
    classifier's attention regions for each detected condition.
    """
    from dataclasses import dataclass as _dc
    from ml.lesion_detector import DetectedLesion, DiseaseFinding
    from ml.pathologies import DISEASE_CATALOG

    @_dc
    class ModelDiseaseFinding(DiseaseFinding):
        model_threshold: Optional[float] = None
        test_auc: Optional[float] = None
        test_sensitivity: Optional[float] = None
        test_specificity: Optional[float] = None
        test_positives: int = 0
        model_name: str = ""

    findings, regions, detected = [], [], []
    for cid, pred in sorted(result.predictions.items(), key=lambda kv: -kv[1].probability):
        d = DISEASE_CATALOG.get(cid)
        if d is None:
            continue
        if pred.detected:
            detected.append(cid)
            regions += attention_regions(result, cid, d.short_name, image_shape,
                                         analysis.valid_retinal_pixels, d.clinical_action)
        findings.append(ModelDiseaseFinding(
            disease_id=cid, disease_name=d.name, acronym=d.acronym, category=d.category,
            is_detected=pred.detected, confidence=round(pred.probability, 4), severity="None",
            hallmark_findings=list(d.hallmark_findings) if pred.detected else [],
            triage_urgency=d.triage_urgency, clinical_action=d.clinical_action, key_biomarkers=[],
            model_threshold=round(pred.threshold, 4), test_auc=pred.test_auc,
            test_sensitivity=pred.test_sensitivity, test_specificity=pred.test_specificity,
            test_positives=pred.test_positives, model_name=result.model_name,
        ))

    normal = DISEASE_CATALOG["normal"]
    is_normal = not detected and not result.abnormal_detected
    findings.insert(0, ModelDiseaseFinding(
        disease_id="normal", disease_name=normal.name, acronym=normal.acronym, category=normal.category,
        is_detected=is_normal, confidence=round(1.0 - result.abnormal_probability, 4), severity="None",
        hallmark_findings=list(normal.hallmark_findings) if is_normal else [],
        triage_urgency=normal.triage_urgency, clinical_action=normal.clinical_action, key_biomarkers=[],
        model_name=result.model_name,
    ))

    if detected:
        primary = max(detected, key=lambda c: result.predictions[c].probability)
        primary_name = DISEASE_CATALOG[primary].name
    elif result.abnormal_detected:
        primary, primary_name = "unspecified_abnormality", "Retinal Abnormality (type not identified)"
    else:
        primary, primary_name = "normal", normal.name

    analysis.disease_findings = findings
    analysis.primary_disease = primary
    analysis.primary_disease_name = primary_name
    analysis.detected_pathologies = [DISEASE_CATALOG[c].acronym for c in detected] or (["Normal"] if is_normal else [])
    analysis.lesions = [DetectedLesion(**r) for r in regions]
    analysis.lesion_count = len(regions)
    analysis.class_summaries = {
        "attention_region": {
            "count": len(regions),
            "area_pixels": int(sum(r["area_pixels"] for r in regions)),
            "affected_area_percentage": round(sum(r["affected_area_percentage"] for r in regions), 3),
        }
    } if regions else {}
    analysis.overall_affected_area_percentage = None  # attention areas are not lesion areas
    analysis.cup_to_disc_ratio = None
