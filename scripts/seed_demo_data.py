"""Demonstration Data Seeder for RetinaLens.

Generates representative clinical screening examinations spanning:
- Healthy Normal (Grade 0)
- Mild NPDR (Grade 1)
- Moderate NPDR (Grade 2 - Referable)
- Severe NPDR (Grade 3 - Referable)
- Proliferative DR (Grade 4 - Urgent Referral)
- Quality-rejected captures (Blur, underexposed)

Populates database, uploads, heatmaps, and printable PDF reports.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.database import SessionLocal, init_db
from backend.models import ScreeningRecord
from backend.service import screening_service
from ml.data import create_synthetic_fundus_image


def seed_demo_screenings(num_samples: int = 15):
    print("Initializing database...")
    init_db()
    db = SessionLocal()

    # Check if database already has records
    existing = db.query(ScreeningRecord).count()
    if existing >= 5:
        print(f"Database already contains {existing} screening records. Skipping seed.")
        db.close()
        return

    print(f"Generating {num_samples} realistic screening cases...")
    random.seed(42)

    patients = [
        {"id": "PAT-7821", "age": 58, "eye": "OD", "grade": 0},
        {"id": "PAT-9314", "age": 64, "eye": "OS", "grade": 1},
        {"id": "PAT-4109", "age": 52, "eye": "OD", "grade": 2},
        {"id": "PAT-5532", "age": 71, "eye": "OD", "grade": 3},
        {"id": "PAT-6681", "age": 69, "eye": "OS", "grade": 4},
        {"id": "PAT-2204", "age": 45, "eye": "OD", "grade": 0},
        {"id": "PAT-3389", "age": 61, "eye": "OS", "grade": 2},
        {"id": "PAT-1190", "age": 55, "eye": "OD", "grade": 1},
        {"id": "PAT-8840", "age": 76, "eye": "OS", "grade": 3},
        {"id": "PAT-9012", "age": 50, "eye": "OD", "grade": 0},
        {"id": "PAT-3411", "age": 63, "eye": "OS", "grade": 2},
        {"id": "PAT-7729", "age": 67, "eye": "OD", "grade": 4},
    ]

    # Process normal cases
    for idx, p in enumerate(patients):
        img_rgb = create_synthetic_fundus_image(grade=p["grade"], size=300, seed=100 + idx)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".png", bgr)
        img_bytes = encoded.tobytes()

        record = screening_service.process_screening(
            db=db,
            image_bytes=img_bytes,
            filename=f"fundus_{p['id']}_{p['eye']}.png",
            patient_id=p["id"],
            patient_age=p["age"],
            eye_side=p["eye"],
        )

        # Distribute dates across the past 10 days for nice trends
        days_ago = random.randint(0, 10)
        record.created_at = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=random.randint(1, 12))
        db.commit()
        print(f"  [+] Created Screening {record.id[:8]} for {p['id']} - Grade {record.grade} ({record.grade_label})")

    # Add 2 quality-rejected cases (defocused blur and dark capture)
    # Case A: Heavy Blur
    blur_img = create_synthetic_fundus_image(grade=2, size=300, seed=999)
    blur_img = cv2.GaussianBlur(blur_img, (31, 31), 10.0)
    _, enc_blur = cv2.imencode(".png", cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR))
    rec_blur = screening_service.process_screening(
        db=db,
        image_bytes=enc_blur.tobytes(),
        filename="blurred_scan.png",
        patient_id="PAT-REJECT-BLUR",
        patient_age=59,
        eye_side="OD",
    )
    rec_blur.created_at = datetime.now(timezone.utc) - timedelta(days=2)
    db.commit()
    print(f"  [!] Created Quality Rejection (Blur): {rec_blur.id[:8]}")

    # Case B: Heavy Underexposure
    dark_img = create_synthetic_fundus_image(grade=1, size=300, seed=888)
    dark_img = (dark_img * 0.15).astype(np.uint8)
    _, enc_dark = cv2.imencode(".png", cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR))
    rec_dark = screening_service.process_screening(
        db=db,
        image_bytes=enc_dark.tobytes(),
        filename="dark_scan.png",
        patient_id="PAT-REJECT-DARK",
        patient_age=68,
        eye_side="OS",
    )
    rec_dark.created_at = datetime.now(timezone.utc) - timedelta(days=4)
    db.commit()
    print(f"  [!] Created Quality Rejection (Underexposed): {rec_dark.id[:8]}")

    total = db.query(ScreeningRecord).count()
    print(f"\nSuccessfully seeded {total} screening records in SQLite database!")
    db.close()


if __name__ == "__main__":
    seed_demo_screenings()
