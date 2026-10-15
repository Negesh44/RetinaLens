"""Build the multi-disease training set from RFMiD and ODIR (Hugging Face parquet mirrors).

Each image is cropped to the circular fundus field, padded to a square and resized to 512 px,
then written to data/multidisease/images/. The manifest maps every image to the RetinaLens
condition ids with 1 = present, 0 = absent, -1 = not annotated by that dataset.

Usage:
    python -m ml.prepare_multidisease
"""

import ast
import glob
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "multidisease"
SIZE = 512

# RFMiD label vector order (official CSV header after ID and Disease_Risk).
RFMID_CODES = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH",
    "ODP", "ODE", "ST", "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA", "VS", "BRAO", "PLQ", "HPED", "CL",
]

# RetinaLens condition id -> RFMiD codes (None = RFMiD does not annotate it) and ODIR label.
CLASSES = {
    "abnormal":                 {"rfmid": "Disease_Risk", "odir": "abnormal", "name": "Any retinal abnormality"},
    "diabetic_retinopathy":     {"rfmid": ["DR"], "odir": "diabetes", "name": "Diabetic Retinopathy"},
    "amd":                      {"rfmid": ["ARMD"], "odir": "age_related_macular_degeneration", "name": "Age-Related Macular Degeneration"},
    "glaucoma":                 {"rfmid": ["ODC"], "odir": "glaucoma", "name": "Glaucoma (optic disc cupping)"},
    "pathologic_myopia":        {"rfmid": ["MYA"], "odir": "pathological_myopia", "name": "Pathologic Myopia"},
    "hypertensive_retinopathy": {"rfmid": None, "odir": "hypertension", "name": "Hypertensive Retinopathy"},
    "vascular_occlusion":       {"rfmid": ["BRVO", "CRVO", "CRAO", "BRAO"], "odir": None, "name": "Retinal Vascular Occlusion"},
    "csc":                      {"rfmid": ["CSR"], "odir": None, "name": "Central Serous Chorioretinopathy"},
    "papilledema":              {"rfmid": ["ODE"], "odir": None, "name": "Optic Disc Edema"},
    "chorioretinitis":          {"rfmid": ["CRS", "RS"], "odir": None, "name": "Chorioretinitis / Retinitis"},
    "macular_hole_pucker":      {"rfmid": ["MHL", "ERM"], "odir": None, "name": "Macular Hole / Epiretinal Membrane"},
    "inherited_dystrophy":      {"rfmid": ["RP"], "odir": None, "name": "Retinitis Pigmentosa"},
}
ODIR_ANNOTATED = {c for c, m in CLASSES.items() if m["odir"]}


def crop_fundus(img: np.ndarray) -> np.ndarray:
    """Crop to the bright circular field of view, pad to a square, resize to SIZE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) > 0.05 * mask.size:
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _write(args):
    data, rel = args
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return rel, False
    cv2.imwrite(str(OUT / rel), crop_fundus(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rel, True


def rfmid_rows():
    import pyarrow.parquet as pq

    rows, jobs = [], []
    for split_hf, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        idx = 0
        for f in sorted(glob.glob(str(ROOT / "data/hf_rfmid/data" / f"{split_hf}-*.parquet"))):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=16, columns=["image", "Disease_Risk", "label"]):
                for rec in batch.to_pylist():
                    vec = dict(zip(RFMID_CODES, ast.literal_eval(rec["label"])))
                    risk = int(rec["Disease_Risk"])
                    rel = f"images/rfmid_{split}_{idx:04d}.jpg"
                    row = {"path": rel, "source": "rfmid", "split": split, "group": f"rfmid_{split}_{idx}"}
                    for c, m in CLASSES.items():
                        if m["rfmid"] == "Disease_Risk":
                            row[c] = risk
                        elif m["rfmid"] is None:
                            row[c] = 0 if risk == 0 else -1  # a normal eye has none of them
                        else:
                            row[c] = int(any(vec[k] for k in m["rfmid"]))
                    rows.append(row)
                    jobs.append((rec["image"]["bytes"], rel))
                    idx += 1
    return rows, jobs


def odir_rows():
    import pyarrow.parquet as pq

    f = glob.glob(str(ROOT / "data/hf_odir/data/*.parquet"))[0]
    meta = pq.read_table(f, columns=["patient_id", "label"]).to_pandas()
    # Split by patient so both eyes of one person stay in the same split.
    patients = np.array(sorted(meta.patient_id.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    n = len(patients)
    split_of = {p: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, p in enumerate(patients)}

    rows, jobs = [], []
    pf = pq.ParquetFile(f)
    idx = 0
    for batch in pf.iter_batches(batch_size=64, columns=["image", "patient_id", "label"]):
        for rec in batch.to_pylist():
            label = rec["label"]
            split = split_of[rec["patient_id"]]
            rel = f"images/odir_{idx:05d}.jpg"
            row = {"path": rel, "source": "odir", "split": split, "group": f"odir_{rec['patient_id']}"}
            for c, m in CLASSES.items():
                if c == "abnormal":
                    row[c] = 0 if label == "normal" else 1
                elif label == "normal":
                    row[c] = 0
                elif c in ODIR_ANNOTATED:
                    row[c] = int(m["odir"] == label)
                else:
                    row[c] = -1
            rows.append(row)
            jobs.append((rec["image"]["bytes"], rel))
            idx += 1
    return rows, jobs


def main():
    import pandas as pd

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for fn in (rfmid_rows, odir_rows):
        r, j = fn()
        rows += r
        jobs += j
        print(fn.__name__, len(r), flush=True)

    ok = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, (rel, good) in enumerate(ex.map(_write, jobs, chunksize=8)):
            ok[rel] = good
            if i % 500 == 0:
                print("written", i, flush=True)
    df = pd.DataFrame(rows)
    df = df[df.path.map(ok)].reset_index(drop=True)
    df.to_csv(OUT / "manifest.csv", index=False)

    classes = list(CLASSES)
    summary = {}
    for split in ("train", "val", "test"):
        d = df[df.split == split]
        summary[split] = {"images": int(len(d)), **{c: int((d[c] == 1).sum()) for c in classes}}
    (OUT / "classes.json").write_text(json.dumps({
        "classes": classes,
        "names": {c: m["name"] for c, m in CLASSES.items()},
        "sources": {"rfmid": "RFMiD (Pachade et al. 2021), CC BY 4.0", "odir": "ODIR-5K (Peking University)"},
        "summary": summary,
    }, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
