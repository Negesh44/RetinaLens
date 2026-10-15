"""RetinaLens FastAPI Application Entrypoint.

Provides:
- POST /api/screen: Image quality triage, AI grading, Grad-CAM, PDF reporting.
- GET  /api/results: Filterable list of previous patient screenings.
- GET  /api/results/{id}: Detailed screening examination record.
- GET  /api/stats: Aggregate epidemiology dashboard KPIs and longitudinal trends.
- GET  /api/report/{id}: Download printable clinical PDF report.
- Static media routes for uploads and heatmaps.
"""

import re
import json
import subprocess
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Query,
    Depends,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from ml.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CLINICAL_DISCLAIMER,
)
import logging
from backend.config import (
    UPLOADS_DIR,
    HEATMAPS_DIR,
    REPORTS_DIR,
    CORS_ORIGINS,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_IMAGE_EXTENSIONS,
)
from backend.database import get_db, init_db
from backend.models import ScreeningRecord
from backend.schemas import (
    ScreeningResponse,
    ScreeningListResponse,
    DashboardStatsResponse,
    ChatRequest,
    ChatResponse,
)
from backend.service import screening_service
from ml.pathologies import DISEASE_CATALOG

logger = logging.getLogger("retinalens.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing database tables and storage."""
    init_db()
    yield


app = FastAPI(
    title=f"{PROJECT_NAME} Clinical AI Screening API",
    version=PROJECT_VERSION,
    description="Explainable AI Screening System for Diabetic Retinopathy from Retinal Fundus Images.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static media directories
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAPS_DIR)), name="heatmaps")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


def _record_to_response(record: ScreeningRecord, base_url: str = "") -> ScreeningResponse:
    """Helper to convert ScreeningRecord DB entity to API Pydantic response."""
    orig_url = None
    if record.original_image_path and Path(record.original_image_path).exists():
        orig_url = f"/uploads/{Path(record.original_image_path).name}"

    heatmap_url = None
    if record.heatmap_image_path and Path(record.heatmap_image_path).exists():
        heatmap_url = f"/heatmaps/{Path(record.heatmap_image_path).name}"

    analyzed_url = None
    if record.analyzed_image_path and Path(record.analyzed_image_path).exists():
        analyzed_url = f"/heatmaps/{Path(record.analyzed_image_path).name}"

    # Reports are generated on request (in the chosen language), so any completed screening has one.
    report_url = f"/api/report/{record.id}" if record.status == "completed" else None

    # Build LesionDTOs
    lesions_list = []
    for l_data in (record.lesions or []):
        try:
            # Ensure segmentation_mask alias is set
            if "segmentation_mask" not in l_data and "polygon" in l_data:
                l_data["segmentation_mask"] = l_data["polygon"]
            lesions_list.append(l_data)
        except Exception:
            pass

    warnings = []
    if record.valid_retinal_pixels and record.total_image_pixels:
        if record.valid_retinal_pixels < (record.total_image_pixels * 0.3):
            warnings.append("Reduced field-of-view coverage detected; affected-area percentage is calculated over the visible retinal disc.")

    img_res = None
    if record.original_image_path and Path(record.original_image_path).exists():
        try:
            import cv2
            img_shape = cv2.imread(record.original_image_path).shape
            img_res = [img_shape[1], img_shape[0]]
        except Exception:
            pass

    # Process multi-disease findings
    disease_findings = record.disease_findings or []
    detected_pathologies = [df["disease_id"] for df in disease_findings if df.get("is_detected")]
    primary_id = record.primary_disease or ("diabetic_retinopathy" if (record.grade and record.grade > 0) else "normal")
    primary_meta = DISEASE_CATALOG.get(primary_id)
    primary_name = primary_meta.name if primary_meta else ("Normal / Physiological Fundus" if primary_id == "normal" else primary_id.replace("_", " ").title())

    return ScreeningResponse(
        id=record.id,
        image_id=record.id,
        created_at=record.created_at,
        patient_id=record.patient_id,
        patient_age=record.patient_age,
        eye_side=record.eye_side or "Unknown",
        status=record.status,
        analysis_status=record.status,
        quality_score=record.quality_score,
        quality_acceptable=record.quality_acceptable,
        quality_reasons=record.quality_reasons,
        grade=record.grade,
        retinal_grade=record.grade,
        grade_label=record.grade_label,
        confidence=record.confidence,
        probabilities=record.probabilities,
        is_referable=record.is_referable,
        uncertain=record.uncertain,
        uncertainty_note=record.uncertainty_note,
        referral_recommendation=record.referral_recommendation,
        lesion_count=record.lesion_count or 0,
        overall_affected_area_percentage=record.overall_affected_area_percentage,
        valid_retinal_pixels=record.valid_retinal_pixels,
        total_image_pixels=record.total_image_pixels,
        calculation_method="Exact pixel-level union mask over segmented valid retinal field",
        image_resolution=img_res,
        class_summaries=record.class_summaries or {},
        lesions=lesions_list,
        warnings=warnings,
        primary_disease=primary_id,
        primary_disease_name=primary_name,
        detected_pathologies=detected_pathologies,
        disease_findings=disease_findings,
        cup_to_disc_ratio=None,
        original_image_url=orig_url,
        analyzed_image_url=analyzed_url,
        heatmap_url=heatmap_url,
        report_download_url=report_url,
        model_version=record.model_version,
        disclaimer=CLINICAL_DISCLAIMER,
    )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
TRAINING_LOG = Path(__file__).resolve().parents[1] / "runs" / "multidisease_train.log"
TRAINING_REPORT = Path(__file__).resolve().parents[1] / "checkpoints" / "retina_multidisease_report.json"
ENSEMBLE_DESCRIPTION = Path(__file__).resolve().parents[1] / "checkpoints" / "retina_ensemble.json"
_EPOCH_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ epoch (\d+)/(\d+) loss ([\d.]+) val mean AUC ([\d.]+) \((\d+)s\)", re.M
)


def _gpu_stats() -> Optional[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        return {"name": name, "utilization": int(util), "memory_used_mb": int(used),
                "memory_total_mb": int(total), "temperature_c": int(temp)}
    except Exception:
        return None


@app.get("/api/training/progress")
def training_progress():
    """Live progress of the multi-disease classifier training run, parsed from its log."""
    if not TRAINING_LOG.exists():
        return {"status": "not_started"}
    text = TRAINING_LOG.read_text(encoding="utf-8", errors="replace")
    epochs = [
        {"epoch": int(m[1]), "loss": float(m[3]), "val_auc": float(m[4]), "seconds": int(m[5]),
         "finished_at": datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S").isoformat()}
        for m in _EPOCH_RE.findall(text)
    ]
    total = int(_EPOCH_RE.findall(text)[-1][2]) if epochs else None
    log_age = time.time() - TRAINING_LOG.stat().st_mtime

    if "Traceback" in text:
        status = "failed"
    elif ".onnx and report" in text:
        status = "completed"
    elif log_age > 20 * 60:
        status = "stopped"
    elif not epochs:
        status = "starting"
    elif epochs[-1]["epoch"] == total:
        status = "evaluating"
    else:
        status = "training"

    best = max(epochs, key=lambda e: e["val_auc"]) if epochs else None
    eta_seconds, epoch_fraction = None, 0.0
    if epochs and status == "training":
        recent = [e["seconds"] for e in epochs[-5:]]
        avg = sum(recent) / len(recent)
        since_last = (datetime.now() - datetime.fromisoformat(epochs[-1]["finished_at"])).total_seconds()
        epoch_fraction = min(0.99, max(0.0, since_last / avg))
        eta_seconds = max(0, int(avg * (total - epochs[-1]["epoch"]) - since_last + 120))

    result = {
        "status": status,
        "total_epochs": total,
        "completed_epochs": epochs[-1]["epoch"] if epochs else 0,
        "current_epoch_fraction": round(epoch_fraction, 3),
        "eta_seconds": eta_seconds,
        "best_epoch": best["epoch"] if best else None,
        "best_val_auc": best["val_auc"] if best else None,
        "epochs": epochs,
        "gpu": _gpu_stats() if status in ("starting", "training", "evaluating") else None,
        "model_loaded": screening_service.disease_classifier is not None,
    }
    # Test results of the model the screening service actually uses (the ensemble once it exists).
    report_path = ENSEMBLE_DESCRIPTION if ENSEMBLE_DESCRIPTION.exists() else TRAINING_REPORT
    if report_path.exists() and (status == "completed" or report_path == ENSEMBLE_DESCRIPTION):
        report = json.loads(report_path.read_text())
        result["test"] = {
            c: {k: v for k, v in m.items() if k in ("n", "n_pos", "auc", "sensitivity", "specificity", "accuracy")}
            for c, m in report["test"].items()
        }
        result["class_names"] = report.get("class_names", {})
        result["excluded_classes"] = report.get("excluded_classes", [])
        result["test_summary"] = report.get("test_summary")
        result["active_model"] = screening_service.disease_classifier.model_name if screening_service.disease_classifier else None
    return result


@app.get("/api/health")
def health_check():
    """Health check and model status probe."""
    return {
        "status": "healthy",
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "onnx_ready": screening_service.onnx_engine is not None,
        "lesion_detector_ready": screening_service.lesion_detector is not None,
        "supported_lesion_classes": [
            "microaneurysm",
            "hemorrhage",
            "hard_exudate",
            "cotton_wool_spot",
        ],
        "calculation_formula": "Affected Area (%) = (Number of lesion pixels / Number of valid retinal pixels) * 100",
        "disclaimer": CLINICAL_DISCLAIMER,
    }


@app.post("/api/analyze-retina", response_model=ScreeningResponse)
async def analyze_retina(
    file: UploadFile = File(..., description="Retinal fundus image upload for interactive lesion analysis"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    eye_side: str = Form("Unknown"),
    db: Session = Depends(get_db),
):
    """Interactive AI-Powered Retina Analysis and Lesion Detection endpoint."""
    return await screen_fundus_image(
        file=file,
        patient_id=patient_id,
        patient_age=patient_age,
        eye_side=eye_side,
        db=db,
    )


@app.post("/api/screen", response_model=ScreeningResponse)
async def screen_fundus_image(
    file: UploadFile = File(..., description="Retinal fundus image upload"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    eye_side: str = Form("Unknown"),
    db: Session = Depends(get_db),
):
    """Performs image quality check, DR grading, calibrated confidence scoring,

    Grad-CAM explainability, and stores the screening examination record.
    """
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {list(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum allowable size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    try:
        record = screening_service.process_screening(
            db=db,
            image_bytes=content,
            filename=file.filename or "upload.png",
            patient_id=patient_id,
            patient_age=patient_age,
            eye_side=eye_side,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Screening processing failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal screening processing error.")

    return _record_to_response(record)


@app.get("/api/results", response_model=ScreeningListResponse)
def list_screening_results(
    grade: Optional[int] = Query(None, ge=0, le=4, description="Filter by DR grade (0-4)"),
    referable_only: Optional[bool] = Query(None, description="Filter for referable cases (Grade >= 2)"),
    patient_id: Optional[str] = Query(None, description="Filter by patient identifier"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieves paginated historical screening examination records with filtering."""
    total, items = screening_service.list_screenings(
        db=db,
        grade=grade,
        referable_only=referable_only,
        patient_id=patient_id,
        skip=skip,
        limit=limit,
    )
    return ScreeningListResponse(
        total=total,
        items=[_record_to_response(r) for r in items],
    )


@app.get("/api/results/{screening_id}", response_model=ScreeningResponse)
def get_screening_result(screening_id: str, db: Session = Depends(get_db)):
    """Retrieves a single screening result record by ID."""
    record = screening_service.get_screening(db, screening_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")
    return _record_to_response(record)


@app.get("/api/stats", response_model=DashboardStatsResponse)
def get_screening_statistics(db: Session = Depends(get_db)):
    """Retrieves aggregate clinical epidemiology stats, referral rates, and daily screening trends."""
    return screening_service.get_dashboard_stats(db)


@app.get("/api/analytics")
def screening_analytics(days: Optional[int] = Query(None, ge=1, le=3650), db: Session = Depends(get_db)):
    """Dashboard analytics: outcomes, follow-up worklist, conditions, trends, age groups and photo quality."""
    from backend.analytics import build_analytics
    from backend.models import ScreeningRecord as Record

    return build_analytics(db.query(Record).all(), days)


@app.get("/api/chat/status")
def chat_status():
    """Whether the report chat assistant is configured (never reveals the key)."""
    from backend.llm import chat_provider

    return {"configured": chat_provider.configured, "provider": chat_provider.name, "model": chat_provider.model}


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_assistant(payload: ChatRequest, db: Session = Depends(get_db)):
    """Answers a question about a screening report using Google Gemini."""
    from backend.chat_assistant import answer
    from backend.llm import ChatProviderError

    record = None
    if payload.screening_id:
        record = screening_service.get_screening(db, payload.screening_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")
    if payload.messages[-1].role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The last message must come from the user.")
    try:
        reply = answer(record, [m.model_dump() for m in payload.messages])
    except ChatProviderError as e:
        logger.warning("Chat assistant unavailable: %s", e.code)
        raise HTTPException(status_code=e.status, detail={"code": e.code, "message": e.message})
    return ChatResponse(reply=reply.text, model=reply.model)


@app.get("/api/report-languages")
def report_languages():
    """Languages the PDF report can be downloaded in."""
    from backend.report_i18n import LANGUAGES

    return [{"code": code, "name": meta["name"], "native": meta["native"]} for code, meta in LANGUAGES.items()]


@app.get("/api/report/{screening_id}")
def download_clinical_report(screening_id: str, lang: str = "en", db: Session = Depends(get_db)):
    """Downloads the PDF screening report in the requested language (default English)."""
    from backend.report_i18n import LANGUAGES
    from backend.report_localized import generate_localized_report

    record = screening_service.get_screening(db, screening_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")
    if lang not in LANGUAGES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported language '{lang}'.")

    pdf_path = generate_localized_report(record, lang)
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (record.patient_id or record.id[:8]))
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"RetinaLens_Report_{name}_{LANGUAGES[lang]['name']}.pdf",
    )
