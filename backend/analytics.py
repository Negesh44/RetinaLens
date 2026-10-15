"""Screening analytics for the Patient Records dashboard.

Condition statistics only use screenings analysed by the trained disease model (their findings
carry a `model_name`); older screenings from the colour-heuristic engine are counted for volume
and photo quality but excluded from condition figures, because their findings are unreliable.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ml.pathologies import DISEASE_CATALOG

URGENCY_ORDER = ["emergency", "urgent", "prompt", "routine"]
# A follow-up is overdue once this many days have passed without a newer screening.
FOLLOW_UP_DAYS = {"emergency": 1, "urgent": 2, "prompt": 28}
AGE_GROUPS = [("Under 40", 0, 39), ("40–49", 40, 49), ("50–59", 50, 59), ("60–69", 60, 69), ("70+", 70, 200)]
QUALITY_ISSUES = [
    ("Blurry / out of focus", ("blur", "defocus", "sharpness")),
    ("Too dark", ("underexposed", "dark")),
    ("Too bright / washed out", ("overexposed", "washout", "saturat")),
    ("Low contrast", ("contrast",)),
    ("Retina not fully in frame", ("field of view", "fov")),
]


def urgency_tier(triage: str) -> str:
    t = (triage or "").lower()
    if "emergency" in t or "same day" in t:
        return "emergency"
    if "urgent" in t or "24" in t:
        return "urgent"
    if "prompt" in t or "wk" in t or "week" in t:
        return "prompt"
    return "routine"


def _local(dt: datetime) -> datetime:
    """Stored timestamps are UTC (naive in SQLite); present them in the server's local time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def summarize(record) -> Dict:
    """Outcome of one screening in dashboard terms."""
    findings = record.disease_findings or []
    from_model = any(f.get("model_name") for f in findings)
    primary_finding = next((f for f in findings if f.get("disease_id") == record.primary_disease), None)
    base = {
        "id": record.id, "patient_id": record.patient_id, "age": record.patient_age, "eye": record.eye_side,
        "date": _local(record.created_at), "from_model": from_model, "quality": record.quality_score,
        "confidence": primary_finding.get("confidence") if primary_finding and record.primary_disease != "normal" else None,
    }
    if record.status == "rejected_for_quality":
        return {**base, "outcome": "rejected", "condition": None, "condition_name": None, "urgency": None, "conditions": []}
    if not from_model:
        return {**base, "outcome": "legacy", "condition": None, "condition_name": None, "urgency": None, "conditions": []}

    detected = [f["disease_id"] for f in findings if f.get("disease_id") != "normal" and f.get("is_detected")]
    primary = record.primary_disease or "normal"
    if primary == "normal" and not detected:
        return {**base, "outcome": "normal", "condition": "normal", "condition_name": "No abnormality", "urgency": "routine", "conditions": []}
    if primary in DISEASE_CATALOG and primary != "normal":
        d = DISEASE_CATALOG[primary]
        return {**base, "outcome": "abnormal", "condition": primary, "condition_name": d.short_name,
                "urgency": urgency_tier(d.triage_urgency), "conditions": detected}
    return {**base, "outcome": "abnormal", "condition": "unspecified_abnormality", "condition_name": "Abnormality (type not identified)",
            "urgency": "prompt", "conditions": detected}


def build_analytics(records: List, days: Optional[int] = None) -> Dict:
    now = datetime.now().astimezone()
    all_rows = [summarize(r) for r in records]
    since = now - timedelta(days=days) if days else None
    rows = [r for r in all_rows if since is None or r["date"] >= since]
    model_rows = [r for r in rows if r["from_model"] and r["outcome"] in ("normal", "abnormal")]
    abnormal = [r for r in model_rows if r["outcome"] == "abnormal"]

    # Condition prevalence (any detected condition, not only the main one).
    prevalence = Counter(c for r in abnormal for c in r["conditions"])
    conditions = [
        {"id": cid, "name": DISEASE_CATALOG[cid].short_name if cid in DISEASE_CATALOG else cid, "count": n,
         "pct": round(100 * n / max(len(model_rows), 1), 1)}
        for cid, n in prevalence.most_common()
    ]

    urgency = Counter(r["urgency"] for r in model_rows)

    # Daily volume across the window (or the full history), by outcome.
    # Start at the first day with screenings (not an empty window), but show at least a week.
    first = min((r["date"].date() for r in rows), default=now.date())
    start = min(first, now.date() - timedelta(days=6))
    if since:
        start = max(start, since.date())
    trend_days = (now.date() - start).days + 1
    by_day = defaultdict(Counter)
    for r in rows:
        by_day[r["date"].date()][r["outcome"]] += 1
    trend = [
        {"date": (start + timedelta(days=i)).isoformat(),
         **{k: by_day[start + timedelta(days=i)][k] for k in ("normal", "abnormal", "rejected", "legacy")}}
        for i in range(trend_days)
    ]

    # Age groups: abnormal rate among model-analysed screenings.
    ages = []
    for label, lo, hi in AGE_GROUPS:
        group = [r for r in model_rows if r["age"] is not None and lo <= r["age"] <= hi]
        n_abn = sum(r["outcome"] == "abnormal" for r in group)
        ages.append({"group": label, "screened": len(group), "abnormal": n_abn,
                     "abnormal_pct": round(100 * n_abn / len(group), 1) if group else None})

    # Photo quality.
    in_window = {r["id"] for r in rows}
    rejected_records = [rec for rec, row in zip(records, all_rows) if rec.id in in_window and row["outcome"] == "rejected"]
    issues = Counter()
    for rec in rejected_records:
        text = " ".join(rec.quality_reasons or []).lower()
        matched = [label for label, keys in QUALITY_ISSUES if any(k in text for k in keys)]
        issues.update(matched or ["Other"])
    scores = [rec.quality_score for rec in records if rec.id in in_window and rec.quality_score is not None]

    # Follow-up worklist: each patient's latest screening that still needs an eye doctor.
    latest: Dict[str, Dict] = {}
    for r in sorted(all_rows, key=lambda r: r["date"]):
        latest[r["patient_id"] or r["id"]] = r
    worklist = []
    for r in latest.values():
        if r["outcome"] == "rejected":
            action, tier = "Retake photo", "retake"
        elif r["outcome"] == "abnormal" and r["urgency"] in FOLLOW_UP_DAYS:
            action, tier = r["condition_name"], r["urgency"]
        else:
            continue
        age_days = (now - r["date"]).days
        due = FOLLOW_UP_DAYS.get(tier, 7)
        worklist.append({
            "screening_id": r["id"], "patient_id": r["patient_id"] or "Anonymous", "age": r["age"], "eye": r["eye"],
            "condition": r["condition"], "action": action, "urgency": tier, "date": r["date"].isoformat(),
            "days_since": age_days, "overdue": age_days > due,
        })
    tier_rank = {"emergency": 0, "urgent": 1, "prompt": 2, "retake": 3}
    worklist.sort(key=lambda w: (not w["overdue"], tier_rank[w["urgency"]], -w["days_since"]))

    patients = {r["patient_id"] for r in rows if r["patient_id"]}
    repeat = Counter(r["patient_id"] for r in all_rows if r["patient_id"])
    return {
        "period_days": days,
        "totals": {
            "screenings": len(rows),
            "patients": len(patients),
            "repeat_patients": sum(1 for p in patients if repeat[p] > 1),
            "analysed_by_model": len(model_rows),
            "normal": len(model_rows) - len(abnormal),
            "abnormal": len(abnormal),
            "abnormal_pct": round(100 * len(abnormal) / max(len(model_rows), 1), 1),
            "rejected": sum(r["outcome"] == "rejected" for r in rows),
            "rejected_pct": round(100 * sum(r["outcome"] == "rejected" for r in rows) / max(len(rows), 1), 1),
            "legacy": sum(r["outcome"] == "legacy" for r in rows),
            "needs_follow_up": sum(1 for w in worklist if w["urgency"] != "retake"),
            "overdue": sum(1 for w in worklist if w["overdue"]),
        },
        "urgency": {k: urgency.get(k, 0) for k in URGENCY_ORDER},
        "conditions": conditions,
        "trend": trend,
        "age_groups": ages,
        "quality": {
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "issues": [{"issue": k, "count": v} for k, v in issues.most_common()],
        },
        "worklist": worklist,
        # Lightweight rows for the records table (newest first); full reports load on demand.
        "screenings": [
            {"id": r["id"], "date": r["date"].isoformat(), "patient_id": r["patient_id"], "age": r["age"], "eye": r["eye"],
             "outcome": r["outcome"], "condition": r["condition"], "condition_name": r["condition_name"], "urgency": r["urgency"],
             "other_conditions": max(0, len(r["conditions"]) - (1 if r["condition"] in r["conditions"] else 0)),
             "confidence": r["confidence"], "quality": r["quality"], "visits": repeat[r["patient_id"]] if r["patient_id"] else 1}
            for r in sorted(rows, key=lambda r: r["date"], reverse=True)
        ],
    }
