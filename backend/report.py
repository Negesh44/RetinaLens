"""Clinical-Grade PDF Screening Report Generator for RetinaLens.

Utilizes ReportLab to generate publication-quality, printable clinical screening reports
complete with patient identifiers, side-by-side fundus and Grad-CAM visualizations,
calibrated probability breakdown, clinical triage recommendations,
and the mandatory regulatory disclaimer.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)

from ml.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CLINICAL_DISCLAIMER,
    ICDR_GRADES,
    ICDR_DESCRIPTIONS,
    REFERABLE_GRADE_THRESHOLD,
)
from backend.config import REPORTS_DIR
from ml.pathologies import DISEASE_CATALOG


def generate_pdf_report(
    screening_record,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Generates a complete printable PDF screening report.

    Args:
        screening_record: Instance of backend.models.ScreeningRecord or dictionary.
        output_path: Custom output file path. Defaults to storage/reports/{id}.pdf.

    Returns:
        Path to the generated PDF document.
    """
    rec_id = getattr(screening_record, "id", None) or screening_record.get("id", "preview")
    if output_path is None:
        output_path = REPORTS_DIR / f"report_{rec_id}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0284c7"),
    )
    style_subtitle = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
    )
    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#b91c1c"),
        alignment=1,  # Center
    )
    style_section = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
    )
    style_body = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    style_bold_body = ParagraphStyle(
        "ReportBoldBody",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph(f"<b>{PROJECT_NAME}</b> | Clinical AI Screening Report", style_title),
            Paragraph(
                f"<b>Model:</b> {getattr(screening_record, 'model_version', 'v1.0')}<br/>"
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style_subtitle,
            ),
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # Mandatory Prominent Disclaimer Banner
    disclaimer_text = f"CRITICAL NOTICE: {CLINICAL_DISCLAIMER}"
    disclaimer_table = Table([[Paragraph(disclaimer_text, style_disclaimer)]], colWidths=[7.2 * inch])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fca5a5")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # 2. Patient & Examination Information
    patient_id = getattr(screening_record, "patient_id", None) or "Anonymous / Unspecified"
    patient_age = getattr(screening_record, "patient_age", None)
    age_str = f"{patient_age} yrs" if patient_age else "Not Recorded"
    eye_side = getattr(screening_record, "eye_side", "Unknown")
    exam_date = getattr(screening_record, "created_at", datetime.now()).strftime("%Y-%m-%d %H:%M UTC")

    patient_info_data = [
        [
            Paragraph("<b>Patient ID:</b>", style_body), Paragraph(str(patient_id), style_bold_body),
            Paragraph("<b>Accession No:</b>", style_body), Paragraph(str(rec_id)[:8].upper(), style_bold_body),
        ],
        [
            Paragraph("<b>Patient Age:</b>", style_body), Paragraph(age_str, style_bold_body),
            Paragraph("<b>Eye Examined:</b>", style_body), Paragraph(str(eye_side), style_bold_body),
        ],
        [
            Paragraph("<b>Exam Date:</b>", style_body), Paragraph(exam_date, style_bold_body),
            Paragraph("<b>Screening Status:</b>", style_body), Paragraph(str(getattr(screening_record, "status", "Completed")).title(), style_bold_body),
        ],
    ]
    patient_table = Table(patient_info_data, colWidths=[1.3 * inch, 2.3 * inch, 1.3 * inch, 2.3 * inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # 3. Image Visualizations: Fundus Scan & Grad-CAM Heatmap
    orig_img_path = getattr(screening_record, "original_image_path", None)
    cam_img_path = getattr(screening_record, "heatmap_image_path", None)

    img_elements = []
    cell_w = 3.4 * inch
    cell_h = 2.4 * inch

    if orig_img_path and Path(orig_img_path).exists():
        img_left = RLImage(str(orig_img_path), width=cell_w, height=cell_h)
    else:
        img_left = Paragraph("<i>Original Fundus scan not archived on disk</i>", style_body)

    if cam_img_path and Path(cam_img_path).exists():
        img_right = RLImage(str(cam_img_path), width=cell_w, height=cell_h)
    else:
        img_right = Paragraph("<i>Grad-CAM heatmap not available</i>", style_body)

    img_table = Table(
        [
            [Paragraph("<b>Primary Fundus Photograph</b>", style_section), Paragraph("<b>Explainable AI Grad-CAM Overlay</b>", style_section)],
            [img_left, img_right],
            [
                Paragraph("<font color='#64748b'>Preprocessed & contrast normalized</font>", style_subtitle),
                Paragraph("<font color='#64748b'>Warm colors highlight neural network attribution</font>", style_subtitle),
            ],
        ],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 4. Screening Assessment & Clinical Recommendation
    grade = getattr(screening_record, "grade", 0)
    grade_int = int(grade) if grade is not None else 0
    grade_desc = ICDR_DESCRIPTIONS.get(grade_int, ICDR_DESCRIPTIONS[0])
    grade_name = ICDR_GRADES.get(grade_int, "Unknown")
    confidence = getattr(screening_record, "confidence", 0.0) or 0.0
    conf_pct = f"{confidence * 100:.1f}%"
    is_referable = bool(getattr(screening_record, "is_referable", False) or grade_int >= REFERABLE_GRADE_THRESHOLD)
    uncertain = bool(getattr(screening_record, "uncertain", False))
    quality_score = float(getattr(screening_record, "quality_score", 1.0) or 1.0)

    # Color code badge
    grade_colors = ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]
    badge_color = grade_colors[min(grade_int, 4)]

    referral_badge = (
        f"<font color='#dc2626'><b>REFERABLE DR DETECTED (Grade &gt;= 2)</b></font>"
        if is_referable else
        f"<font color='#16a34a'><b>NON-REFERABLE (Grade &lt; 2)</b></font>"
    )

    assessment_data = [
        [
            Paragraph("<b>Predicted ICDR Severity:</b>", style_body),
            Paragraph(f"<font color='{badge_color}'><b>Grade {grade_int}: {grade_name}</b></font>", style_bold_body),
            Paragraph("<b>Calibrated Confidence:</b>", style_body),
            Paragraph(conf_pct, style_bold_body),
        ],
        [
            Paragraph("<b>Clinical Triage Flag:</b>", style_body),
            Paragraph(referral_badge, style_body),
            Paragraph("<b>Image Quality Score:</b>", style_body),
            Paragraph(f"{quality_score:.2f} / 1.00", style_bold_body),
        ],
        [
            Paragraph("<b>Recommended Action:</b>", style_body),
            Paragraph(f"<b>{grade_desc['action']}</b>", style_bold_body),
            Paragraph("<b>Key Pathologic Signs:</b>", style_body),
            Paragraph(grade_desc["findings"], style_body),
        ],
    ]

    lesion_count = int(getattr(screening_record, "lesion_count", 0) or 0)
    affected_pct = getattr(screening_record, "overall_affected_area_percentage", None)
    affected_str = f"{affected_pct:.2f}% of retina" if affected_pct is not None else "Not available"

    assessment_data.append([
        Paragraph("<b>Detected Lesions:</b>", style_body),
        Paragraph(f"<b>{lesion_count}</b> focal lesions ({affected_str})", style_bold_body),
        Paragraph("<b>Quantification:</b>", style_body),
        Paragraph("Non-overlapping union mask", style_body),
    ])

    primary_id = getattr(screening_record, "primary_disease", None)
    if primary_id and primary_id != "normal":
        prim_meta = DISEASE_CATALOG.get(primary_id)
        prim_name = getattr(prim_meta, "name", primary_id.replace("_", " ").title()) if prim_meta else primary_id.replace("_", " ").title()
        prim_urgency = getattr(prim_meta, "triage_urgency", "High") if prim_meta else "High"
        assessment_data.append([
            Paragraph("<b>Primary Pathology:</b>", style_body),
            Paragraph(f"<font color='#0284c7'><b>{prim_name}</b></font>", style_bold_body),
            Paragraph("<b>Diagnostic Urgency:</b>", style_body),
            Paragraph(f"<b>{prim_urgency}</b>", style_bold_body),
        ])


    if uncertain:
        assessment_data.append([
            Paragraph("<b>Uncertainty Flag:</b>", style_body),
            Paragraph("<font color='#b45309'><b>Uncertain Prediction: Specialist ophthalmic review strongly advised</b></font>", style_body),
            Paragraph("", style_body), Paragraph("", style_body),
        ])

    assessment_table = Table(assessment_data, colWidths=[1.6 * inch, 2.2 * inch, 1.4 * inch, 2.0 * inch])
    assessment_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("<b>Screening Findings & Clinical Triage</b>", style_section))
    story.append(Spacer(1, 4))
    story.append(assessment_table)
    story.append(Spacer(1, 8))

    # Multi-Condition Retinal Pathology Differential Table
    raw_findings = getattr(screening_record, "disease_findings", []) or []
    detected_findings = [f for f in raw_findings if f.get("is_detected") or f.get("confidence", 0) >= 0.4]
    if detected_findings:
        diff_rows = [
            [
                Paragraph("<b>Condition / Acronym</b>", style_subtitle),
                Paragraph("<b>Severity</b>", style_subtitle),
                Paragraph("<b>Urgency</b>", style_subtitle),
                Paragraph("<b>Identified Clinical Hallmarks</b>", style_subtitle),
            ]
        ]
        for df in detected_findings:
            c_name = df.get("disease_name", df.get("disease_id", "Unknown"))
            c_acro = df.get("acronym", "")
            c_sev = df.get("severity", "Moderate")
            c_urg = df.get("triage_urgency", "Routine")
            c_hallmarks = ", ".join(df.get("hallmark_findings", [])) or "Biomarkers detected"
            
            urg_str = f"<font color='#b91c1c'><b>{c_urg}</b></font>" if ("Urgent" in c_urg or "Emergency" in c_urg) else c_urg
            diff_rows.append([
                Paragraph(f"<b>{c_name}</b> ({c_acro})", style_bold_body),
                Paragraph(c_sev, style_body),
                Paragraph(urg_str, style_body),
                Paragraph(c_hallmarks, style_body),
            ])

        diff_table = Table(diff_rows, colWidths=[2.2 * inch, 1.1 * inch, 1.5 * inch, 2.4 * inch])
        diff_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("<b>Multi-Condition Pathology Differential Assessment</b>", style_section))
        story.append(Spacer(1, 4))
        story.append(diff_table)
        story.append(Spacer(1, 8))


    # 5. Calibrated Probabilities Table
    probs = getattr(screening_record, "probabilities", {}) or {}
    prob_headers = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Proliferative)"]
    prob_values = []
    for g in range(5):
        val = probs.get(str(g), probs.get(g, 0.0))
        prob_values.append(f"{val * 100:.1f}%")

    prob_data = [
        [Paragraph(f"<b>{h}</b>", style_subtitle) for h in prob_headers],
        [Paragraph(f"<b>{v}</b>", style_bold_body) for v in prob_values],
    ]
    prob_table = Table(prob_data, colWidths=[1.44 * inch] * 5)
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("<b>Calibrated Severity Probability Distribution</b>", style_section))
    story.append(Spacer(1, 4))
    story.append(prob_table)
    story.append(Spacer(1, 12))

    # 6. Clinician Review & Sign-Off Section
    sign_off_data = [
        [
            Paragraph("<b>Reviewing Clinician:</b> ___________________________", style_body),
            Paragraph("<b>Clinical Concurrence:</b> [  ] Agree  [  ] Disagree  [  ] Inconclusive", style_body),
        ],
        [
            Paragraph("<b>Clinician Signature:</b> ___________________________", style_body),
            Paragraph("<b>Review Date:</b> ___________________________", style_body),
        ],
    ]
    sign_table = Table(sign_off_data, colWidths=[3.6 * inch, 3.6 * inch])
    sign_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("<b>Clinician Sign-off & Audit Trail</b>", style_section))
    story.append(Spacer(1, 4))
    story.append(sign_table)

    # Build PDF
    doc.build(story)
    return output_path
