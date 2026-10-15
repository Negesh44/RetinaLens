"""Tests for the Patient Records analytics (backend.analytics)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.analytics import build_analytics

NOW = datetime.now(timezone.utc).replace(tzinfo=None)  # stored timestamps are naive UTC


def rec(rid, patient, days_ago=0, status="completed", primary="normal", detected=(), model=True, age=None, reasons=()):
    findings = [{"disease_id": "normal", "is_detected": not detected, "confidence": 0.9, "model_name": "m" if model else ""}]
    findings += [{"disease_id": d, "is_detected": True, "confidence": 0.8, "model_name": "m" if model else ""} for d in detected]
    return SimpleNamespace(id=rid, patient_id=patient, patient_age=age, eye_side="OD", status=status,
                           created_at=NOW - timedelta(days=days_ago), quality_score=0.9, quality_reasons=list(reasons),
                           primary_disease=primary if status == "completed" else "unassessable",
                           disease_findings=findings if status == "completed" else [])


def test_totals_conditions_and_urgency():
    records = [
        rec("1", "P1", primary="glaucoma", detected=["glaucoma", "amd"], age=62),
        rec("2", "P2", primary="normal", age=45),
        rec("3", "P3", primary="papilledema", detected=["papilledema"], age=55),
        rec("4", "P4", status="rejected_for_quality", reasons=["Significant blur / optical defocus detected"]),
        rec("5", "P5", primary="amd", detected=["amd"], model=False),  # older engine
    ]
    a = build_analytics(records)
    t = a["totals"]
    assert (t["screenings"], t["analysed_by_model"], t["abnormal"], t["normal"], t["rejected"], t["legacy"]) == (5, 3, 2, 1, 1, 1)
    assert {c["id"]: c["count"] for c in a["conditions"]} == {"glaucoma": 1, "amd": 1, "papilledema": 1}  # legacy AMD excluded
    assert a["urgency"] == {"emergency": 1, "urgent": 0, "prompt": 1, "routine": 1}
    assert a["quality"]["issues"] == [{"issue": "Blurry / out of focus", "count": 1}]
    ages = {g["group"]: g for g in a["age_groups"]}
    assert ages["60–69"]["abnormal_pct"] == 100.0 and ages["40–49"]["abnormal_pct"] == 0.0


def test_worklist_uses_latest_screening_and_flags_overdue():
    records = [
        rec("old", "P1", days_ago=40, primary="glaucoma", detected=["glaucoma"]),
        rec("new", "P1", days_ago=1, primary="normal"),  # newer normal visit clears P1
        rec("late", "P2", days_ago=35, primary="glaucoma", detected=["glaucoma"]),  # prompt, > 28 days
        rec("er", "P3", days_ago=0, primary="papilledema", detected=["papilledema"]),
        rec("blur", "P4", days_ago=2, status="rejected_for_quality"),
    ]
    w = build_analytics(records)["worklist"]
    assert [x["screening_id"] for x in w] == ["late", "er", "blur"]  # overdue first, then by urgency
    assert w[0]["overdue"] and not w[1]["overdue"]
    assert w[2]["urgency"] == "retake"


def test_period_filter_and_table_rows():
    records = [rec("recent", "P1", days_ago=2), rec("old", "P2", days_ago=60)]
    a = build_analytics(records, days=30)
    assert a["totals"]["screenings"] == 1
    assert [r["id"] for r in a["screenings"]] == ["recent"]
    assert len(a["trend"]) == 7  # starts at the first screening day, padded to a week
