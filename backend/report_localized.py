"""Patient screening report as a PDF, in English or an Indian language.

Text comes from backend.report_i18n. Indian scripts are shaped with HarfBuzz (uharfbuzz), so
conjuncts and vowel signs display correctly. Each language uses one font that also contains the
Latin glyphs (built by scripts/build_report_fonts.py), because ReportLab shapes a word with a
single font and a word mixing scripts would otherwise render wrongly.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.config import REPORTS_DIR
from backend.report_i18n import LANGUAGES, TEXT

FONT_DIR = Path(__file__).resolve().parent / "fonts"
SCRIPT_FONTS = {
    "latin": "NotoSans",
    "devanagari": "RetinaLensReportDevanagari",
    "bengali": "RetinaLensReportBengali",
    "tamil": "RetinaLensReportTamil",
    "telugu": "RetinaLensReportTelugu",
    "kannada": "RetinaLensReportKannada",
    "malayalam": "RetinaLensReportMalayalam",
}
# Same colours as the web app (frontend/src/lib/diseases.js).
DISEASE_COLORS = {
    "normal": "#10b981", "diabetic_retinopathy": "#38bdf8", "amd": "#f59e0b", "glaucoma": "#22d3ee",
    "pathologic_myopia": "#0d9488", "hypertensive_retinopathy": "#fb923c", "vascular_occlusion": "#818cf8",
    "csc": "#2dd4bf", "papilledema": "#dc2626", "chorioretinitis": "#8b5cf6", "macular_hole_pucker": "#d946ef",
}
PURPLE = colors.HexColor("#3b2463")
INK = colors.HexColor("#231d1c")
MUTED = colors.HexColor("#6b5e52")
SAND = colors.HexColor("#f6efe3")
LINE = colors.HexColor("#e3d7c3")

_registered = False


def _register_fonts():
    global _registered
    if _registered:
        return
    for family in SCRIPT_FONTS.values():
        for weight in ("Regular", "Bold"):
            pdfmetrics.registerFont(TTFont(f"{family}-{weight}", str(FONT_DIR / f"{family}-{weight}.ttf")))
    _registered = True


def _rich(text: str, script: str, bold: bool = False) -> str:
    """Paragraph markup for text in the given script (one font covers its Latin characters too)."""
    return f'<font name="{SCRIPT_FONTS[script]}-{"Bold" if bold else "Regular"}">{escape(text)}</font>'


def _style(name, size, leading=None, color=INK, align=0):
    return ParagraphStyle(name, fontName="NotoSans-Regular", fontSize=size, leading=leading or size * 1.45,
                          textColor=color, alignment=align, shaping=1)


def _urgency_tier(triage: str) -> str:
    t = (triage or "").lower()
    if "emergency" in t or "same day" in t:
        return "emergency"
    if "urgent" in t or "24" in t:
        return "urgent"
    if "prompt" in t or "wk" in t or "week" in t:
        return "prompt"
    return "routine"


def _image(path, width):
    if not path or not Path(path).exists():
        return None
    img = Image(str(path))
    scale = width / img.imageWidth
    img.drawWidth, img.drawHeight = width, img.imageHeight * scale
    if img.drawHeight > width:  # keep tall photos within a square box
        factor = width / img.drawHeight
        img.drawWidth, img.drawHeight = img.drawWidth * factor, width
    return img


def generate_localized_report(record, lang: str = "en", output_path: Optional[Union[str, Path]] = None) -> Path:
    if lang not in TEXT:
        lang = "en"
    _register_fonts()
    T = TEXT[lang]
    script = LANGUAGES[lang]["script"]
    R = lambda key, bold=False, **fmt: _rich(T[key].format(**fmt) if fmt else T[key], script, bold)  # noqa: E731
    r = lambda text, bold=False: _rich(text, script, bold)  # noqa: E731

    output_path = Path(output_path or REPORTS_DIR / f"report_{record.id}_{lang}.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=30, bottomMargin=30,
                            title=f"RetinaLens report {record.id[:8]}", author="RetinaLens")
    width = A4[0] - 72

    s_title = _style("t", 17, color=colors.white)
    s_sub = _style("s", 9, color=colors.HexColor("#d6ccea"))
    s_label = _style("l", 8, color=MUTED)
    s_value = _style("v", 9.5)
    s_h = _style("h", 11.5, leading=16, color=PURPLE)
    s_big = _style("b", 16, leading=22)
    s_body = _style("p", 9.5, leading=14.5)
    s_small = _style("sm", 8, leading=11.5, color=MUTED)
    s_doc = ParagraphStyle("doc", parent=s_small, textColor=INK)

    story = []

    # Header band
    header = Table([[Paragraph(R("title", True), s_title), Paragraph('<font name="NotoSans-Bold">RetinaLens</font>', _style("r", 12, color=colors.white, align=2))],
                    [Paragraph(R("subtitle"), s_sub), ""]],
                   colWidths=[width * 0.72, width * 0.28])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
    ]))
    story += [header, Spacer(1, 10)]

    # Patient details
    created = record.created_at or datetime.now()
    age = f"{record.patient_age} {T['years']}" if record.patient_age else T["not_given"]
    eye = T.get(f"eye_{record.eye_side or 'Unknown'}", T["eye_Unknown"])
    cells = [(T["patient_id"], record.patient_id or "-"), (T["age"], age), (T["eye"], eye),
             (T["date"], created.strftime("%d-%m-%Y")), (T["report_id"], record.id[:8].upper()), ("", "")]
    # Three wide columns so long words wrap between words, never inside a letter cluster.
    rows = []
    for i in range(0, len(cells), 3):
        group = cells[i:i + 3]
        rows += [[Paragraph(r(k), s_label) for k, _ in group], [Paragraph(r(v, True), s_value) for _, v in group]]
    details = Table(rows, colWidths=[width / 3] * 3)
    details.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SAND), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, 0), 6), ("TOPPADDING", (0, 2), (-1, 2), 6), ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
    ]))
    story += [details, Spacer(1, 12)]

    findings = record.disease_findings or []
    if record.status == "rejected_for_quality":
        box = Table([[Paragraph(R("rejected_title", True), s_big)], [Paragraph(R("rejected_text"), s_body)]], colWidths=[width])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#b45309")), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                 ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10)]))
        story += [box, Spacer(1, 12)]
    else:
        primary = record.primary_disease or "normal"
        detected = sorted([f for f in findings if f.get("disease_id") != "normal" and f.get("is_detected")],
                          key=lambda f: -f.get("confidence", 0))
        main = next((f for f in findings if f.get("disease_id") == primary), None)
        normal_f = next((f for f in findings if f.get("disease_id") == "normal"), None)

        if primary == "normal":
            title, meaning, todo, tier = T["normal_title"], T["normal_meaning"], T["normal_todo"], "routine"
            conf = normal_f.get("confidence") if normal_f else None
        elif primary in T["conditions"]:
            title, meaning, advice = T["conditions"][primary]
            tier = _urgency_tier(main.get("triage_urgency") if main else "")
            todo = f"{T['urgency'][tier][1]} {advice}"
            conf = main.get("confidence") if main else None
        else:
            title, meaning, todo, tier = T["unspecified_title"], T["unspecified_meaning"], T["unspecified_todo"], "prompt"
            conf = (1 - normal_f["confidence"]) if normal_f else None
        color = colors.HexColor(DISEASE_COLORS.get(primary, "#f59e0b"))
        urgency_color = {"emergency": "#dc2626", "urgent": "#e11d48", "prompt": "#b45309", "routine": "#047857"}[tier]

        rows = [
            [Paragraph(R("main_finding"), s_label)],
            [Paragraph(r(title, True), s_big)],
            # Keep each label's colon inside the label's own font run (see module docstring).
            [Paragraph(f'{r(T["when"] + ":")} <font color="{urgency_color}">{r(T["urgency"][tier][0], True)}</font>', s_body)],
            *([[Paragraph(f'{r(T["confidence"] + ":")} {r(f"{round(conf * 100)}%", True)}', s_body)]] if conf is not None else []),
            [Paragraph(f'{R("meaning", True)}<br/>{r(meaning)}', s_body)],
            [Paragraph(f'{R("todo", True)}<br/>{r(todo)}', s_body)],
        ]
        if main and main.get("test_sensitivity") is not None and primary != "normal":
            rows.append([Paragraph(R("reliability", n=main.get("test_positives", 0),
                                     sens=round(main["test_sensitivity"] * 100),
                                     spec=round((main.get("test_specificity") or 0) * 100)), s_small)])
        result = Table(rows, colWidths=[width - 6])
        result.setStyle(TableStyle([
            ("LINEBEFORE", (0, 0), (0, -1), 5, color), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, -1), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, 0), 10),
        ]))
        story += [Paragraph(R("result", True), s_h), Spacer(1, 4), result, Spacer(1, 10)]

        others = [f for f in detected if f.get("disease_id") != primary]
        if primary != "normal":
            story.append(Paragraph(R("also_found", True), s_h))
            if others:
                for f in others:
                    name = T["conditions"].get(f["disease_id"], [f.get("disease_name", f["disease_id"])])[0]
                    story.append(Paragraph(f'• {r(name, True)} — {r(str(round(f.get("confidence", 0) * 100)) + "%")}', s_body))
            else:
                story.append(Paragraph(R("none_also"), s_body))
            story.append(Spacer(1, 10))

    # Photos: original and AI attention areas
    img_w = (width - 16) / 2
    left = _image(record.original_image_path, img_w) or Paragraph("-", s_small)
    has_regions = (record.lesion_count or 0) > 0 and record.status != "rejected_for_quality"
    right = (_image(record.analyzed_image_path, img_w) if has_regions else None) or Paragraph(R("no_attention"), s_small)
    photos = Table([[Paragraph(R("photo", True), s_label), Paragraph(R("attention", True), s_label)], [left, right]],
                   colWidths=[img_w + 8, img_w + 8])
    photos.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 1), (-1, 1), "CENTER"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, 0), 4)]))
    story += [KeepTogether([photos]), Spacer(1, 12)]

    # Emergency symptoms and limits
    emergency = Table([[Paragraph(f'{R("emergency_title", True)}<br/>{R("emergency_text")}', s_body)]], colWidths=[width])
    emergency.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdecec")),
                                   ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f2a7a7")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [KeepTogether([emergency]), Spacer(1, 8),
              Paragraph(R("not_covered"), s_small), Spacer(1, 3), Paragraph(R("disclaimer", True), s_small), Spacer(1, 12)]

    # Clinician section (always English)
    if findings and record.status != "rejected_for_quality":
        rows = [[Paragraph(f'<font name="NotoSans-Bold">{h}</font>', s_doc) for h in ("Condition", "Probability", "Threshold", "Result")]]
        for f in sorted([f for f in findings if f.get("disease_id") != "normal"], key=lambda f: -f.get("confidence", 0)):
            th = f.get("model_threshold")
            rows.append([Paragraph(escape(f.get("disease_name", f["disease_id"])), s_doc),
                         Paragraph(f'{f.get("confidence", 0) * 100:.1f}%', s_doc),
                         Paragraph(f"{th * 100:.0f}%" if th is not None else "-", s_doc),
                         Paragraph('<font name="NotoSans-Bold">Detected</font>' if f.get("is_detected") else "Not detected", s_doc)])
        table = Table(rows, colWidths=[width * 0.52, width * 0.16, width * 0.14, width * 0.18], repeatRows=1)
        table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE), ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
        story += [Paragraph('<font name="NotoSans-Bold">For the eye doctor (English)</font>', s_h),
                  Spacer(1, 4), table, Spacer(1, 16)]

    story.append(Paragraph(R("signature"), s_small))
    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(f"RetinaLens · {LANGUAGES[lang]['name']} · generated {datetime.now():%d-%m-%Y %H:%M}"), s_small))

    doc.build(story)
    return output_path
