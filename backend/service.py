"""Core Inference, Explainability, and Screening Business Logic Service."""

import io
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from dataclasses import asdict
from ml.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CLINICAL_DISCLAIMER,
    ICDR_GRADES,
    ICDR_DESCRIPTIONS,
    REFERABLE_GRADE_THRESHOLD,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_UNCERTAINTY_THRESHOLD,
)
from ml.preprocessing import preprocess_fundus_image, to_model_input
from ml.quality import ImageQualityAssessor
from ml.grading import continuous_to_probabilities
from ml.onnx_engine import ONNXRetinaEngine
from ml.lesion_detector import RetinaLesionDetector
from ml.disease_classifier import ClassifierResult, RetinaDiseaseClassifier, apply_to_analysis, cam_full_frame

from backend.config import (
    UPLOADS_DIR,
    HEATMAPS_DIR,
    REPORTS_DIR,
    CHECKPOINTS_DIR,
    DEFAULT_ONNX_PATH,
    DEFAULT_PYTORCH_PATH,
)
from backend.models import ScreeningRecord
from backend.report_localized import generate_localized_report

logger = logging.getLogger("retinalens.service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ScreeningService:
    """Singleton service managing ML models, ONNX engine, XAI, and persistence."""

    def __init__(self):
        self.quality_assessor = ImageQualityAssessor(min_quality_threshold=DEFAULT_QUALITY_THRESHOLD)
        self.lesion_detector = RetinaLesionDetector()
        # Trained multi-label disease classifier; when present it replaces the heuristic disease scoring.
        self.disease_classifier: Optional[RetinaDiseaseClassifier] = None
        if RetinaDiseaseClassifier.available():
            try:
                self.disease_classifier = RetinaDiseaseClassifier()
            except Exception as e:
                logger.warning("Could not load disease classifier (%s); using heuristic disease scoring.", e)
        self.uncertainty_threshold = DEFAULT_UNCERTAINTY_THRESHOLD
        self.image_size = 224
        self.thresholds = [0.5, 1.5, 2.5, 3.5]
        self.temperature = 1.0

        self.onnx_engine: Optional[ONNXRetinaEngine] = None
        # PyTorch is only loaded when the exported ONNX model is missing (it needs ~1.5 GB more RAM).
        self.pytorch_model = None
        self.explainer = None

        self._initialize_models()

    def _initialize_models(self):
        """Uses the exported ONNX model when present; otherwise builds it with PyTorch."""
        if DEFAULT_ONNX_PATH.exists():
            try:
                self._load_onnx_engine()
                return
            except Exception as e:
                logger.warning("Could not initialize ONNX engine (%s), using PyTorch engine.", e)
        self._initialize_pytorch()

    def _load_onnx_engine(self):
        self.onnx_engine = ONNXRetinaEngine(DEFAULT_ONNX_PATH)
        self.thresholds = self.onnx_engine.metadata.get("thresholds", self.thresholds)
        self.image_size = self.onnx_engine.metadata.get("image_size", self.image_size)
        self.temperature = self.onnx_engine.metadata.get("temperature", 1.0)
        logger.info("ONNX runtime inference engine ready.")

    def _initialize_pytorch(self):
        """PyTorch fallback: the DR model, a one-off ONNX export and the Grad-CAM explainer."""
        import torch
        from ml.export import export_retina_model_to_onnx
        from ml.gradcam import FundusExplainer
        from ml.models import create_retina_model

        self.device = torch.device("cpu")
        # 1. Initialize PyTorch model (used for Grad-CAM and fallback)
        self.pytorch_model = create_retina_model(
            backbone_name="efficientnet_b0",
            pretrained=True,
            head_type="regression",
        )
        self.pytorch_model.eval()

        # If trained weights exist on disk, load them
        if DEFAULT_PYTORCH_PATH.exists():
            try:
                ckpt = torch.load(str(DEFAULT_PYTORCH_PATH), map_location=self.device)
                self.pytorch_model.load_state_dict(ckpt["model_state_dict"])
                self.thresholds = ckpt.get("thresholds", self.thresholds)
                self.image_size = ckpt.get("image_size", self.image_size)
                logger.info("Loaded PyTorch weights from %s", DEFAULT_PYTORCH_PATH)
            except Exception as e:
                logger.warning("Could not load weights from %s: %s", DEFAULT_PYTORCH_PATH, e)

        # 2. Initialize or Export ONNX model
        if not DEFAULT_ONNX_PATH.exists():
            logger.info("Exporting initial ONNX model to %s...", DEFAULT_ONNX_PATH)
            try:
                export_retina_model_to_onnx(
                    model=self.pytorch_model,
                    save_path=DEFAULT_ONNX_PATH,
                    image_size=self.image_size,
                    thresholds=self.thresholds,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error("Failed to export ONNX model: %s", e)

        # 3. Instantiate ONNX inference engine
        if DEFAULT_ONNX_PATH.exists():
            try:
                self._load_onnx_engine()
            except Exception as e:
                logger.warning("Could not initialize ONNX engine (%s), using PyTorch engine.", e)

        # 4. Instantiate Grad-CAM Explainer
        try:
            self.explainer = FundusExplainer(self.pytorch_model, method="gradcam", device=self.device)
            logger.info("Grad-CAM explainability module ready.")
        except Exception as e:
            logger.error("Failed to initialize Grad-CAM explainer: %s", e)

    def process_screening(
        self,
        db: Session,
        image_bytes: bytes,
        filename: str,
        patient_id: Optional[str] = None,
        patient_age: Optional[int] = None,
        eye_side: str = "Unknown",
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> ScreeningRecord:
        """Executes full screening workflow:

        1. Decode raw image.
        2. Quality gate check -> Reject if ungradable.
        3. Preprocessing (border crop + local contrast enhancement).
        4. ONNX forward inference.
        5. Calibrated confidence & uncertainty evaluation.
        6. Grad-CAM attribution generation.
        7. PDF report generation and DB persistence.
        """
        screening_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower() if filename else ".png"
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            ext = ".png"

        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("Invalid or corrupted image format.")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 1. Image Quality Assessment Gate
        q_result = self.quality_assessor.assess_quality(rgb_image, threshold=quality_threshold)
        is_acceptable = q_result["is_acceptable"]
        quality_score = q_result["quality_score"]
        reasons = q_result["reasons"]

        # Save original uploaded image
        orig_filename = f"{screening_id}_orig{ext}"
        orig_file_path = UPLOADS_DIR / orig_filename
        cv2.imwrite(str(orig_file_path), bgr_image)

        # If image fails quality gate, refuse to grade and return retake advice
        if not is_acceptable:
            record = ScreeningRecord(
                id=screening_id,
                created_at=datetime.now(timezone.utc),
                patient_id=patient_id,
                patient_age=patient_age,
                eye_side=eye_side,
                status="rejected_for_quality",
                quality_score=quality_score,
                quality_acceptable=False,
                quality_reasons_json=json.dumps(reasons),
                grade=None,
                grade_label=None,
                confidence=None,
                probabilities_json="{}",
                is_referable=False,
                uncertain=True,
                uncertainty_note="Screening rejected due to insufficient retinal image quality.",
                referral_recommendation=q_result["recommendation"],
                lesion_count=0,
                overall_affected_area_percentage=None,
                valid_retinal_pixels=None,
                total_image_pixels=int(bgr_image.shape[0] * bgr_image.shape[1]),
                lesions_json="[]",
                class_summaries_json="{}",
                primary_disease="unassessable",
                disease_findings_json="[]",
                original_image_path=str(orig_file_path),
                analyzed_image_path=None,
                heatmap_image_path=None,
                raw_heatmap_path=None,
                pdf_report_path=None,
                model_version=f"{PROJECT_NAME}-v{PROJECT_VERSION}",
                disclaimer=CLINICAL_DISCLAIMER,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

        # 2. Quality passed -> Preprocessing
        preprocessed_rgb = preprocess_fundus_image(
            rgb_image,
            target_size=self.image_size,
            use_ben_graham=True,
            crop_border=True,
        )

        model_input = to_model_input(preprocessed_rgb)

        # 3. Fast ONNX Inference (PyTorch only when the ONNX model is missing)
        if self.onnx_engine is not None:
            continuous_pred = float(np.squeeze(self.onnx_engine.predict(model_input)))
        else:
            continuous_pred = self._pytorch_predict(model_input)

        # 4. Discretization using optimized thresholds
        th = self.thresholds
        grade = 0
        if continuous_pred >= th[0]: grade = 1
        if continuous_pred >= th[1]: grade = 2
        if continuous_pred >= th[2]: grade = 3
        if continuous_pred >= th[3]: grade = 4

        grade_label = ICDR_GRADES[grade]
        is_referable = bool(grade >= REFERABLE_GRADE_THRESHOLD)

        # 5. Calibrated Probabilities & Uncertainty
        raw_probs = continuous_to_probabilities(continuous_pred, sigma=0.65)
        # Apply temperature scaling
        cal_logits = np.log(np.clip(raw_probs, 1e-7, 1.0 - 1e-7)) / max(self.temperature, 0.1)
        exp_l = np.exp(cal_logits - np.max(cal_logits))
        calibrated_probs = exp_l / np.sum(exp_l)

        top_confidence = float(calibrated_probs[grade])
        uncertain = bool(top_confidence < self.uncertainty_threshold)
        uncertainty_note = (
            "Confidence below 60%: Uncertain prediction. Specialist clinical review recommended."
            if uncertain else None
        )

        probs_dict = {str(i): round(float(calibrated_probs[i]), 4) for i in range(5)}
        grade_desc = ICDR_DESCRIPTIONS.get(grade, ICDR_DESCRIPTIONS[0])
        referral_rec = grade_desc["action"]

        # 6. Trained multi-disease classifier (its activation maps also drive the heatmap)
        classifier_result: Optional[ClassifierResult] = None
        if self.disease_classifier is not None:
            try:
                classifier_result = self.disease_classifier.predict(rgb_image)
            except Exception as e:
                logger.error("Disease classifier failed (%s); keeping heuristic findings.", e)

        # 7. Explainability heatmap
        cam_filename = f"{screening_id}_cam.png"
        cam_file_path = HEATMAPS_DIR / cam_filename
        raw_cam_filename = f"{screening_id}_raw_cam.npy"
        raw_cam_path = HEATMAPS_DIR / raw_cam_filename

        try:
            if classifier_result is not None:
                raw_heatmap, overlay_rgb = self._classifier_heatmap(classifier_result, rgb_image)
            elif self.explainer is not None:
                import torch
                raw_heatmap, overlay_rgb = self.explainer.explain(
                    input_tensor=torch.from_numpy(model_input),
                    original_rgb=preprocessed_rgb,
                    target_category=None,
                    colormap=cv2.COLORMAP_TURBO,
                    blend_alpha=0.55,
                )
            else:
                raise RuntimeError("no explainability source available")
            # Save overlay image
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cam_file_path), overlay_bgr)
            # Save raw heatmap
            np.save(str(raw_cam_path), raw_heatmap)
        except Exception as e:
            logger.error("Failed to generate Grad-CAM visualization: %s", e)
            cam_file_path = None
            raw_cam_path = None

        # 8. Interactive Lesion Segmentation & Affected Area Quantification
        lesion_analysis = self.lesion_detector.detect_lesions(rgb_image, icdr_grade=grade)
        if classifier_result is not None:
            try:
                apply_to_analysis(classifier_result, lesion_analysis, rgb_image.shape)
            except Exception as e:
                logger.error("Disease classifier failed (%s); keeping heuristic findings.", e)
        analyzed_filename = f"{screening_id}_analyzed.png"
        analyzed_file_path = HEATMAPS_DIR / analyzed_filename

        try:
            analyzed_overlay = self.lesion_detector.render_overlay_image(rgb_image, lesion_analysis)
            cv2.imwrite(str(analyzed_file_path), cv2.cvtColor(analyzed_overlay, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.error("Failed to render lesion overlay image: %s", e)
            analyzed_file_path = None

        # 8. Create Database Record
        record = ScreeningRecord(
            id=screening_id,
            created_at=datetime.now(timezone.utc),
            patient_id=patient_id,
            patient_age=patient_age,
            eye_side=eye_side,
            status="completed",
            quality_score=quality_score,
            quality_acceptable=True,
            quality_reasons_json=json.dumps(reasons),
            grade=grade,
            grade_label=grade_label,
            confidence=round(top_confidence, 4),
            probabilities_json=json.dumps(probs_dict),
            is_referable=is_referable,
            uncertain=uncertain,
            uncertainty_note=uncertainty_note,
            referral_recommendation=referral_rec,
            lesion_count=lesion_analysis.lesion_count,
            overall_affected_area_percentage=lesion_analysis.overall_affected_area_percentage,
            valid_retinal_pixels=lesion_analysis.valid_retinal_pixels,
            total_image_pixels=lesion_analysis.total_image_pixels,
            lesions_json=json.dumps([asdict(l) for l in lesion_analysis.lesions]),
            class_summaries_json=json.dumps(lesion_analysis.class_summaries),
            primary_disease=lesion_analysis.primary_disease,
            disease_findings_json=json.dumps([asdict(df) for df in lesion_analysis.disease_findings]),
            original_image_path=str(orig_file_path),
            analyzed_image_path=str(analyzed_file_path) if analyzed_file_path else None,
            heatmap_image_path=str(cam_file_path) if cam_file_path else None,
            raw_heatmap_path=str(raw_cam_path) if raw_cam_path else None,
            pdf_report_path=None,
            model_version=f"{PROJECT_NAME}-v{PROJECT_VERSION}",
            disclaimer=CLINICAL_DISCLAIMER,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        # 8. Pre-generate printable PDF report
        try:
            pdf_path = generate_localized_report(record, "en")
            record.pdf_report_path = str(pdf_path)
            db.commit()
        except Exception as e:
            logger.error("Failed to generate PDF report: %s", e)

        return record

    def _pytorch_predict(self, model_input: np.ndarray) -> float:
        import torch
        with torch.no_grad():
            return float(self.pytorch_model(torch.from_numpy(model_input)).squeeze().item())

    @staticmethod
    def _classifier_heatmap(result: ClassifierResult, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Activation map of the most likely detected condition (else of 'abnormal'), blended onto the photo."""
        ranked = sorted(result.predictions.items(), key=lambda kv: -kv[1].probability)
        detected = [cid for cid, pred in ranked if pred.detected]
        heatmap = cam_full_frame(result, detected[0] if detected else "abnormal", rgb_image.shape)
        if heatmap is None:
            raise RuntimeError("activation map could not be placed on the image")
        colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_image, 0.45, colored, 0.55, 0)
        return heatmap, overlay

    def get_screening(self, db: Session, screening_id: str) -> Optional[ScreeningRecord]:
        return db.query(ScreeningRecord).filter(ScreeningRecord.id == screening_id).first()

    def list_screenings(
        self,
        db: Session,
        grade: Optional[int] = None,
        referable_only: Optional[bool] = None,
        patient_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[int, List[ScreeningRecord]]:
        query = db.query(ScreeningRecord)

        if grade is not None:
            query = query.filter(ScreeningRecord.grade == grade)
        if referable_only is True:
            query = query.filter(ScreeningRecord.is_referable == True)
        if patient_id:
            query = query.filter(ScreeningRecord.patient_id.ilike(f"%{patient_id}%"))

        total = query.count()
        items = query.order_by(ScreeningRecord.created_at.desc()).offset(skip).limit(limit).all()
        return total, items

    def get_dashboard_stats(self, db: Session) -> Dict:
        """Calculates clinical dashboard metrics and historical screening volume."""
        total_screenings = db.query(ScreeningRecord).count()
        completed = db.query(ScreeningRecord).filter(ScreeningRecord.status == "completed").count()
        rejected = db.query(ScreeningRecord).filter(ScreeningRecord.status == "rejected_for_quality").count()
        referable = db.query(ScreeningRecord).filter(ScreeningRecord.is_referable == True).count()

        referral_rate = (referable / max(completed, 1)) * 100.0

        # Mean confidence of completed screenings
        mean_conf = db.query(func.avg(ScreeningRecord.confidence)).filter(ScreeningRecord.status == "completed").scalar()
        mean_conf_val = round(float(mean_conf), 4) if mean_conf is not None else 0.0

        # Grade distribution
        grade_dist = {i: 0 for i in range(5)}
        for grade_val, count in (
            db.query(ScreeningRecord.grade, func.count(ScreeningRecord.id))
            .filter(ScreeningRecord.status == "completed")
            .group_by(ScreeningRecord.grade)
            .all()
        ):
            if grade_val is not None and 0 <= grade_val <= 4:
                grade_dist[int(grade_val)] = count

        # Daily Trends (last 14 days)
        daily_trends = []
        today = datetime.now(timezone.utc).date()
        for i in range(13, -1, -1):
            day = today - timedelta(days=i)
            day_start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
            day_end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)

            day_total = db.query(ScreeningRecord).filter(
                ScreeningRecord.created_at >= day_start,
                ScreeningRecord.created_at <= day_end,
            ).count()

            day_referable = db.query(ScreeningRecord).filter(
                ScreeningRecord.created_at >= day_start,
                ScreeningRecord.created_at <= day_end,
                ScreeningRecord.is_referable == True,
            ).count()

            day_rejected = db.query(ScreeningRecord).filter(
                ScreeningRecord.created_at >= day_start,
                ScreeningRecord.created_at <= day_end,
                ScreeningRecord.status == "rejected_for_quality",
            ).count()

            daily_trends.append({
                "date": day.strftime("%Y-%m-%d"),
                "total": day_total,
                "referable": day_referable,
                "rejected": day_rejected,
            })

        return {
            "total_screenings": total_screenings,
            "completed_screenings": completed,
            "rejected_for_quality_count": rejected,
            "referable_count": referable,
            "referral_rate_pct": round(referral_rate, 2),
            "mean_confidence": mean_conf_val,
            "grade_distribution": grade_dist,
            "grade_labels": ICDR_GRADES,
            "daily_trends": daily_trends,
        }


# Singleton service instance
screening_service = ScreeningService()
