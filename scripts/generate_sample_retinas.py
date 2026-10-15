import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ml.data import create_synthetic_fundus_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "frontend" / "public" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_samples():
    print(f"Generating clinical fundus samples in {SAMPLES_DIR}...")

    # 1. Normal (Grade 0)
    img0 = create_synthetic_fundus_image(grade=0, size=380, seed=101)
    cv2.imwrite(str(SAMPLES_DIR / "sample_normal_grade0.png"), cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))

    # 2. Mild NPDR (Grade 1)
    img1 = create_synthetic_fundus_image(grade=1, size=380, seed=102)
    cv2.imwrite(str(SAMPLES_DIR / "sample_mild_grade1.png"), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

    # 3. Moderate NPDR (Grade 2)
    img2 = create_synthetic_fundus_image(grade=2, size=380, seed=103)
    cv2.imwrite(str(SAMPLES_DIR / "sample_moderate_grade2.png"), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    # 4. Severe NPDR (Grade 3)
    img3 = create_synthetic_fundus_image(grade=3, size=380, seed=104)
    cv2.imwrite(str(SAMPLES_DIR / "sample_severe_grade3.png"), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))

    # 5. Proliferative DR (Grade 4)
    img4 = create_synthetic_fundus_image(grade=4, size=380, seed=105)
    cv2.imwrite(str(SAMPLES_DIR / "sample_proliferative_grade4.png"), cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))

    # 6. Quality Rejected - Blur
    blur_img = create_synthetic_fundus_image(grade=1, size=380, seed=106)
    blur_img = cv2.GaussianBlur(blur_img, (35, 35), 12.0)
    cv2.imwrite(str(SAMPLES_DIR / "sample_blur_reject.png"), cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR))

    # 7. Quality Rejected - Dark
    dark_img = create_synthetic_fundus_image(grade=2, size=380, seed=107)
    dark_img = (dark_img * 0.12).astype(np.uint8)
    cv2.imwrite(str(SAMPLES_DIR / "sample_underexposed_reject.png"), cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR))

    print("All sample retinal images successfully generated!")


if __name__ == "__main__":
    generate_samples()
