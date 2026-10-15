"""Dataset Management and Loading for Retinal Fundus Screening.

Supports APTOS 2019, EyePACS, IDRiD, Messidor-2, and synthetic fundus generation.
Standardizes all labels to the ICDR 0-4 scale:
0: No DR
1: Mild
2: Moderate
3: Severe
4: Proliferative
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from ml.config import RANDOM_SEED, DATA_DIR, ICDR_GRADES
from ml.preprocessing import (
    crop_retina_border,
    apply_ben_graham_enhancement,
    preprocess_fundus_image,
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger("retinalens.data")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RetinaFundusDataset(Dataset):
    """PyTorch Dataset for retinal fundus images with preprocessing & augmentations."""

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        target_size: int = 224,
        use_ben_graham: bool = True,
        transforms=None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.target_size = target_size
        self.use_ben_graham = use_ben_graham
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocess_fundus_image(
            image,
            target_size=self.target_size,
            use_ben_graham=self.use_ben_graham,
            crop_border=True,
        )

        if self.transforms:
            augmented = self.transforms(image=processed)
            tensor_img = augmented["image"]
        else:
            val_tf = get_val_transforms(self.target_size)
            tensor_img = val_tf(image=processed)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_img, label


def create_synthetic_fundus_image(
    grade: int,
    size: int = 300,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generates a realistic synthetic retinal fundus phantom image.

    Includes:
    - Circular orange-red fundus disc with vascular shading.
    - Yellowish-orange optic nerve head (optic disc).
    - Radiating retinal blood vessels.
    - Pathologic lesions synthesized according to ICDR grade:
        * Grade 0: Normal fundus
        * Grade 1: Few tiny microaneurysms (small red dots)
        * Grade 2: More microaneurysms + small blot hemorrhages + hard exudates (bright yellow flecks)
        * Grade 3: Extensive multi-quadrant hemorrhages + cotton wool spots (soft whitish clouds)
        * Grade 4: Florid neovascularization fronds + severe preretinal hemorrhage
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    # 1. Background retinal fundus gradient (orange-red)
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    fundus_mask = dist_from_center <= radius

    # Base retina color
    red_val = np.clip(180 - (dist_from_center / radius * 50) + rng.normal(0, 3, (size, size)), 80, 210)
    green_val = np.clip(75 - (dist_from_center / radius * 35) + rng.normal(0, 3, (size, size)), 20, 110)
    blue_val = np.clip(25 - (dist_from_center / radius * 15) + rng.normal(0, 2, (size, size)), 5, 45)

    img[fundus_mask, 0] = red_val[fundus_mask].astype(np.uint8)
    img[fundus_mask, 1] = green_val[fundus_mask].astype(np.uint8)
    img[fundus_mask, 2] = blue_val[fundus_mask].astype(np.uint8)

    # 2. Optic Disc (yellowish circle located nasal/temporal)
    disc_center = (int(center[0] - radius * 0.42), int(center[1] + radius * 0.05))
    disc_radius = int(radius * 0.16)
    cv2.circle(img, disc_center, disc_radius, (255, 220, 130), -1)
    cv2.GaussianBlur(img, (7, 7), 2, dst=img)

    # 3. Main blood vessels radiating from optic disc
    for angle_deg in [30, 70, 110, 150, 210, 250, 290, 330]:
        rad = np.radians(angle_deg)
        pts = [disc_center]
        cur_x, cur_y = disc_center
        length = int(radius * 0.8)
        step = 8
        for i in range(0, length, step):
            cur_x += int(step * np.cos(rad) + rng.randint(-2, 3))
            cur_y += int(step * np.sin(rad) + rng.randint(-2, 3))
            if np.sqrt((cur_x - center[0]) ** 2 + (cur_y - center[1]) ** 2) < radius - 5:
                pts.append((cur_x, cur_y))
        if len(pts) > 1:
            thickness = max(1, int(size * 0.008))
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(img, p1, p2, (100, 20, 15), thickness)

    # 4. Synthesize lesions according to grade
    if grade >= 1:
        # Microaneurysms (tiny red dots: r=1-2)
        num_ma = 4 if grade == 1 else (15 if grade == 2 else 35)
        for _ in range(num_ma):
            offset_r = rng.uniform(radius * 0.1, radius * 0.8)
            theta = rng.uniform(0, 2 * np.pi)
            lx = int(center[0] + offset_r * np.cos(theta))
            ly = int(center[1] + offset_r * np.sin(theta))
            cv2.circle(img, (lx, ly), rng.randint(1, 3), (120, 10, 10), -1)

    if grade >= 2:
        # Hard exudates (bright yellow flecks)
        num_ex = 12 if grade == 2 else (30 if grade == 3 else 45)
        for _ in range(num_ex):
            offset_r = rng.uniform(radius * 0.2, radius * 0.75)
            theta = rng.uniform(0, 2 * np.pi)
            lx = int(center[0] + offset_r * np.cos(theta))
            ly = int(center[1] + offset_r * np.sin(theta))
            cv2.circle(img, (lx, ly), rng.randint(2, 4), (250, 240, 140), -1)

        # Blot hemorrhages (dark red irregular spots)
        num_hem = 6 if grade == 2 else (25 if grade == 3 else 40)
        for _ in range(num_hem):
            offset_r = rng.uniform(radius * 0.2, radius * 0.8)
            theta = rng.uniform(0, 2 * np.pi)
            lx = int(center[0] + offset_r * np.cos(theta))
            ly = int(center[1] + offset_r * np.sin(theta))
            cv2.circle(img, (lx, ly), rng.randint(3, 7), (80, 5, 5), -1)

    if grade >= 3:
        # Cotton wool spots (soft whitish lesions)
        for _ in range(6):
            offset_r = rng.uniform(radius * 0.25, radius * 0.7)
            theta = rng.uniform(0, 2 * np.pi)
            lx = int(center[0] + offset_r * np.cos(theta))
            ly = int(center[1] + offset_r * np.sin(theta))
            cv2.circle(img, (lx, ly), rng.randint(6, 12), (230, 220, 210), -1)

    if grade >= 4:
        # Neovascularization & flame/preretinal hemorrhage (large irregular dark blotch)
        hx = int(center[0] + rng.uniform(-0.3, 0.3) * radius)
        hy = int(center[1] + rng.uniform(-0.3, 0.3) * radius)
        cv2.ellipse(img, (hx, hy), (int(radius * 0.25), int(radius * 0.12)), 45, 0, 360, (70, 0, 0), -1)

    # Blur slightly for natural fundus appearance
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    # Ensure outside mask is pitch black
    img[~fundus_mask] = 0
    return img


def generate_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generates a complete synthetic fundus benchmark dataset.

    Returns DataFrame with columns: image_path, grade, grade_name.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    # Typical clinical distribution: more 0s and 1s, fewer severe/proliferative
    probs = [0.40, 0.25, 0.18, 0.10, 0.07]
    grades = rng.choice([0, 1, 2, 3, 4], size=num_samples, p=probs)

    records = []
    for idx, grade in enumerate(grades):
        img_id = f"synth_{idx:04d}"
        img_filename = f"{img_id}.png"
        img_filepath = images_dir / img_filename

        synth_img = create_synthetic_fundus_image(grade=grade, size=320, seed=seed + idx)
        # Save as BGR for OpenCV imwrite
        bgr_img = cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_filepath), bgr_img)

        records.append({
            "image_id": img_id,
            "image_path": str(img_filepath.resolve()),
            "grade": int(grade),
            "grade_name": ICDR_GRADES[grade],
        })

    df = pd.DataFrame(records)
    csv_path = out_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Generated %d synthetic fundus samples at %s", len(df), out_path)
    return df


def load_dataset_metadata(
    dataset_name: str,
    base_dir: Union[str, Path] = DATA_DIR,
) -> Optional[pd.DataFrame]:
    """Loads and standardizes metadata for a specific dataset into columns:

    [image_path, grade, dataset_name].
    Supports: aptos, eyepacs, idrid, messidor2, synthetic.
    Returns None with warning if dataset directory is not present.
    """
    base_path = Path(base_dir)
    target_dir = base_path / dataset_name.lower()

    if not target_dir.exists():
        logger.warning(
            "Dataset '%s' directory not found at '%s'. Skipping dataset.",
            dataset_name,
            target_dir,
        )
        return None

    dataset_key = dataset_name.lower()

    if dataset_key in ["aptos", "aptos2019"]:
        csv_file = target_dir / "train.csv"
        img_dir = target_dir / "train_images"
        if not csv_file.exists():
            logger.warning("APTOS train.csv not found in %s", target_dir)
            return None
        df = pd.read_csv(csv_file)
        # Standardize APTOS columns: id_code -> image_path, diagnosis -> grade
        df["grade"] = df["diagnosis"].astype(int)
        df["image_path"] = df["id_code"].apply(
            lambda x: str((img_dir / f"{x}.png").resolve())
        )
        df["dataset_name"] = "APTOS-2019"

    elif dataset_key in ["eyepacs"]:
        csv_file = target_dir / "trainLabels.csv"
        img_dir = target_dir / "train"
        if not csv_file.exists():
            logger.warning("EyePACS trainLabels.csv not found in %s", target_dir)
            return None
        df = pd.read_csv(csv_file)
        df["grade"] = df["level"].astype(int)
        df["image_path"] = df["image"].apply(
            lambda x: str((img_dir / f"{x}.jpeg").resolve())
        )
        df["dataset_name"] = "EyePACS"

    elif dataset_key in ["idrid"]:
        csv_file = target_dir / "IDRiD_disease_grading.csv"
        img_dir = target_dir / "images"
        if not csv_file.exists():
            logger.warning("IDRiD CSV not found in %s", target_dir)
            return None
        df = pd.read_csv(csv_file)
        # IDRiD has 'Retinopathy grade'
        col_grade = [c for c in df.columns if "grade" in c.lower()][0]
        col_id = [c for c in df.columns if "name" in c.lower() or "id" in c.lower()][0]
        df["grade"] = df[col_grade].astype(int)
        df["image_path"] = df[col_id].apply(
            lambda x: str((img_dir / f"{x}.jpg").resolve())
        )
        df["dataset_name"] = "IDRiD"

    elif dataset_key in ["messidor", "messidor2"]:
        csv_file = target_dir / "messidor_data.csv"
        img_dir = target_dir / "images"
        if not csv_file.exists():
            logger.warning("Messidor-2 CSV not found in %s", target_dir)
            return None
        df = pd.read_csv(csv_file)
        # Messidor-2 adjudicated grade 0-4
        col_grade = [c for c in df.columns if "grade" in c.lower() or "level" in c.lower()][0]
        col_id = [c for c in df.columns if "id" in c.lower() or "image" in c.lower()][0]
        df["grade"] = df[col_grade].astype(int)
        df["image_path"] = df[col_id].apply(
            lambda x: str((img_dir / f"{x}").resolve())
        )
        df["dataset_name"] = "Messidor-2"

    elif dataset_key in ["synthetic", "synth"]:
        csv_file = target_dir / "labels.csv"
        if not csv_file.exists():
            df = generate_synthetic_dataset(target_dir, num_samples=100)
        else:
            df = pd.read_csv(csv_file)
        df["dataset_name"] = "Synthetic-Fundus"

    else:
        logger.warning("Unknown dataset name '%s'", dataset_name)
        return None

    # Filter out entries where file does not exist
    initial_len = len(df)
    df = df[df["image_path"].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    logger.info(
        "Loaded %d valid images for %s (initially listed %d)",
        len(df),
        dataset_name,
        initial_len,
    )
    return df[["image_path", "grade", "dataset_name"]]


def prepare_stratified_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into stratified train and validation sets by DR grade."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["grade"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_class_distribution_report(df: pd.DataFrame) -> Dict[str, Union[int, Dict[int, int]]]:
    """Generates distribution counts and percentage breakdown by grade."""
    counts = df["grade"].value_counts().to_dict()
    total = len(df)
    distribution = {
        "total_samples": total,
        "counts": {g: counts.get(g, 0) for g in range(5)},
        "percentages": {g: round((counts.get(g, 0) / max(total, 1)) * 100, 2) for g in range(5)},
        "referable_count": sum(counts.get(g, 0) for g in [2, 3, 4]),
        "referable_percentage": round((sum(counts.get(g, 0) for g in [2, 3, 4]) / max(total, 1)) * 100, 2),
    }
    return distribution


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler to balance class frequencies during training."""
    class_counts = np.bincount(labels, minlength=5)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler
