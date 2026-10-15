"""Patient-friendly chat assistant grounded in a RetinaLens screening report.

Builds the system prompt from the screening record (findings, urgency, measured reliability)
and the disease catalogue, then asks the chat provider (backend.llm) for an answer.
"""

from typing import Dict, List, Optional

from backend.llm import ChatReply, chat_provider
from ml.pathologies import DISEASE_CATALOG

MAX_TURNS = 12  # most recent messages sent to the model
MAX_MESSAGE_CHARS = 2000

GUIDELINES = """You are the RetinaLens eye-health assistant. You help patients and families understand a retina screening report in a kind, calm and simple way.

How to answer:
- Use plain, everyday words. Explain any medical term the first time you use it.
- Keep answers short: a few sentences or a short bullet list, unless the person asks for more detail.
- Be warm and reassuring, but honest. Never frighten, never give false hope.
- Reply in the same language the person writes in (for example Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi or English).

What you must follow:
- Base everything about THIS patient only on the report below. Do not invent findings, measurements or history.
- This is an AI screening result, not a diagnosis. Say so when it matters, and always guide the person to an eye doctor (ophthalmologist) to confirm.
- Stage or severity: RetinaLens does not measure the stage of a disease. If asked about the stage, say clearly that the screening cannot tell the stage, explain in simple words what the usual stages of that disease are, and say an eye doctor finds the stage with a dilated eye examination and tests such as OCT. Never guess a stage.
- Use the report's urgency ("when to see an eye doctor") for timing advice.
- If the person mentions sudden loss of vision, a curtain or shadow over their sight, many new floaters or flashes of light, or severe eye pain, tell them to go to an eye emergency service immediately.
- Do not prescribe medicines or doses, and do not tell anyone to stop a treatment. General healthy advice (blood sugar and blood pressure control, not smoking, regular check-ups) is fine.
- If a question is not about eyes, health or this report, politely bring the conversation back to the report.
"""


def _pct(value: Optional[float]) -> str:
    return f"{round(value * 100)}%" if value is not None else "unknown"


def report_context(record) -> str:
    """Plain-text summary of a screening record for the model."""
    if record is None:
        return "No screening report is open. Answer general questions about retina and eye health, and suggest uploading a retina photo for a screening."

    lines = [
        "SCREENING REPORT",
        f"- Date: {record.created_at:%d %B %Y}" if record.created_at else "- Date: unknown",
        f"- Patient age: {record.patient_age} years" if record.patient_age else "- Patient age: not given",
        f"- Eye: {({'OD': 'right eye', 'OS': 'left eye'}).get(record.eye_side, 'not specified')}",
    ]
    if record.status == "rejected_for_quality":
        lines.append("- Result: the photo was not clear enough to analyse. The patient should have the retina photo retaken.")
        return "\n".join(lines)

    findings = record.disease_findings or []
    detected = sorted([f for f in findings if f.get("disease_id") != "normal" and f.get("is_detected")],
                      key=lambda f: -f.get("confidence", 0))
    normal = next((f for f in findings if f.get("disease_id") == "normal"), None)
    primary = record.primary_disease or "normal"

    if primary == "normal" and not detected:
        lines.append("- Result: no signs of the screened eye diseases were found (normal screening).")
        if normal:
            lines.append(f"- Probability that any disease is present: {_pct(1 - normal.get('confidence', 0))}")
    elif primary == "unspecified_abnormality":
        lines.append("- Result: the retina looks abnormal, but the AI could not tell which disease. A full eye examination is advised within a few weeks.")
    for i, f in enumerate(detected):
        d = DISEASE_CATALOG.get(f["disease_id"])
        label = "Main finding" if i == 0 else "Also found"
        lines.append(f"- {label}: {d.name if d else f.get('disease_name')} (AI probability {_pct(f.get('confidence'))})")
        if d:
            lines.append(f"  - What it is: {d.summary}")
            lines.append(f"  - When to see an eye doctor: {d.triage_urgency}")
            lines.append(f"  - Usual next steps: {d.clinical_action}")
        if f.get("test_sensitivity") is not None:
            lines.append(f"  - How reliable the AI is for this condition (tested on {f.get('test_positives', 0)} eyes with it): "
                         f"it found {_pct(f['test_sensitivity'])} of cases and correctly cleared {_pct(f.get('test_specificity'))} of eyes without it.")

    screened = [DISEASE_CATALOG[f["disease_id"]].short_name for f in findings
                if f.get("disease_id") in DISEASE_CATALOG and f.get("disease_id") != "normal"]
    if screened:
        lines.append(f"- Conditions this screening checks: {', '.join(screened)}.")
    lines.append("- Not checked by this AI: retinal detachment, eye tumours, diabetic macular oedema, retinitis pigmentosa.")
    return "\n".join(lines)


def answer(record, messages: List[Dict[str, str]]) -> ChatReply:
    """Answer the latest user message in the conversation, grounded in the report."""
    clean = [
        {"role": m["role"], "content": str(m["content"])[:MAX_MESSAGE_CHARS]}
        for m in messages
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ][-MAX_TURNS:]
    while clean and clean[0]["role"] != "user":  # a conversation must start with the user
        clean.pop(0)
    system = f"{GUIDELINES}\n\n{report_context(record)}"
    return chat_provider.chat(clean, system)
