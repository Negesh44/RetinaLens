"""Train the multi-label retinal disease classifier on real fundus photographs.

Reads the manifest written by ``ml.prepare_multidisease`` (one row per cached 512 px image,
one column per condition with 1 = present, 0 = absent, -1 = not annotated by that dataset).
Unannotated labels are masked out of the loss, so a dataset that never labels a condition
does not teach the model that it is absent.

Usage (GPU environment):
    .venv-train/Scripts/python.exe -m ml.train_multidisease --epochs 30
"""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "multidisease"
OUT_DIR = ROOT / "checkpoints"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

logger = logging.getLogger("retinalens.multidisease")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def build_transforms(size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
            A.HueSaturationValue(5, 12, 8, p=0.4),
            A.RandomGamma(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes, transform):
        self.paths = df["path"].tolist()
        self.labels = df[classes].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(DATA_DIR / self.paths[i])), cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], torch.from_numpy(self.labels[i])


class CamExportWrapper(nn.Module):
    """Returns logits plus a class activation map per class.

    The head is global-average-pool followed by a linear layer, so projecting the final
    feature map through the classifier weights gives the exact spatial contribution to each logit.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)
        logits = self.model.forward_head(features)
        cams = torch.einsum("kc,bchw->bkhw", self.model.get_classifier().weight, features)
        return logits, cams


def masked_bce(logits, targets, pos_weight):
    mask = (targets >= 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets.clamp(min=0), pos_weight=pos_weight, reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(x)
            if tta:
                logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def per_class_metrics(probs, labels, classes, thresholds=None):
    """AUC, sensitivity, specificity, accuracy per class over annotated samples only."""
    out = {}
    for j, c in enumerate(classes):
        m = labels[:, j] >= 0
        y, p = labels[m, j], probs[m, j]
        n_pos = int(y.sum())
        entry = {"n": int(m.sum()), "n_pos": n_pos}
        if 0 < n_pos < len(y):
            entry["auc"] = float(roc_auc_score(y, p))
        if thresholds is not None and len(y):
            pred = p >= thresholds[c]
            tp = int((pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
            fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
            entry.update({
                "threshold": float(thresholds[c]),
                "sensitivity": tp / (tp + fn) if tp + fn else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
                "accuracy": (tp + tn) / len(y),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })
        out[c] = entry
    return out


def tune_thresholds(probs, labels, classes, min_specificity=0.95):
    """Per-class threshold chosen on validation data only.

    Takes the lowest threshold that still clears at least `min_specificity` of eyes without the
    condition (maximising sensitivity under that constraint), so rare conditions do not produce a
    stream of weak false alarms. Falls back to Youden's J if the constraint cannot be met.
    """
    th = {}
    for j, c in enumerate(classes):
        m = labels[:, j] >= 0
        y, p = labels[m, j], probs[m, j]
        if y.sum() == 0 or y.sum() == len(y):
            th[c] = 0.5
            continue
        neg = p[y == 0]
        cand = float(np.quantile(neg, min_specificity)) + 1e-6
        if (p[y == 1] >= cand).mean() > 0:
            th[c] = min(cand, 0.99)
            continue
        best, best_t = -1.0, 0.5
        for t in np.unique(np.round(p, 3)):
            pred = p >= t
            sens = (pred & (y == 1)).sum() / y.sum()
            spec = (~pred & (y == 0)).sum() / (y == 0).sum()
            if sens + spec - 1 > best:
                best, best_t = sens + spec - 1, float(t)
        th[c] = best_t
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b3.ra2_in1k")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay for weight averaging (0 = off)")
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="retina_multidisease")
    ap.add_argument("--min-positives", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    ap.add_argument("--min-specificity", type=float, default=0.90,
                    help="decision thresholds keep at least this specificity on validation data")
    ap.add_argument("--min-probability", type=float, default=0.20,
                    help="floor applied to every decision threshold")
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip training; tune thresholds, test and export the saved best checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((DATA_DIR / "classes.json").read_text())
    classes = meta["classes"]
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if args.limit:
        tr, va, te = (d.sample(min(len(d), args.limit), random_state=0).reset_index(drop=True) for d in (tr, va, te))
    # A condition needs enough positive training images to be learned (and evaluated) meaningfully.
    excluded = [c for c in classes if (tr[c] == 1).sum() < args.min_positives]
    classes = [c for c in classes if c not in excluded]
    if excluded:
        logger.info("excluding conditions with < %d training positives: %s", args.min_positives, excluded)
    logger.info("device=%s train=%d val=%d test=%d classes=%d", device, len(tr), len(va), len(te), len(classes))

    # Positive weights from annotated training labels, capped so rare classes do not dominate.
    pos = (tr[classes] == 1).sum().to_numpy(); neg = (tr[classes] == 0).sum().to_numpy()
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 15.0), dtype=torch.float32, device=device)

    # Oversample images carrying a rare positive label.
    freq = {c: max(1, int((tr[c] == 1).sum())) for c in classes}
    weights = tr.apply(lambda r: max([1.0] + [min(8.0, 200.0 / freq[c]) for c in classes if r[c] == 1]), axis=1)
    sampler = torch.utils.data.WeightedRandomSampler(weights.to_numpy(), num_samples=len(tr), replacement=True)

    mk = lambda d, train: DataLoader(
        FundusDataset(d, classes, build_transforms(args.size, train)),
        batch_size=args.batch if train else args.batch * 2, sampler=sampler if train else None,
        shuffle=False, num_workers=args.workers if train else min(args.workers, 3), pin_memory=True, drop_last=train,
        # Only the training loader keeps its workers alive. On Windows every worker loads the CUDA
        # libraries, and too many live workers exhaust the system commit limit.
        persistent_workers=train and args.workers > 0,
    )
    train_dl, val_dl, test_dl = mk(tr, True), mk(va, False), mk(te, False)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(classes), drop_rate=0.3,
                              drop_path_rate=args.drop_path).to(device)
    # Exponential moving average of the weights: a smoother model that usually generalises better.
    ema = timm.utils.ModelEmaV3(model, decay=args.ema) if args.ema > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl); warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_auc, best_path = -1.0, OUT_DIR / f"{args.name}_best.pt"
    history = []
    if args.finalize_only:
        # Recover the per-epoch history from the training log of the interrupted run.
        log = ROOT / "runs" / f"{args.name.replace('retina_', '')}_train.log"
        if log.exists():
            for m in re.finditer(r"epoch (\d+)/\d+ loss ([\d.]+) val mean AUC ([\d.]+)", log.read_text(errors="replace")):
                history.append({"epoch": int(m[1]), "loss": float(m[2]), "val_mean_auc": float(m[3])})
        best_auc = max((h["val_mean_auc"] for h in history), default=float("nan"))
    else:
        for epoch in range(1, args.epochs + 1):
            model.train(); t0 = time.time(); running = 0.0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = masked_bce(model(x), y, pos_weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update(); sched.step()
                if ema is not None:
                    ema.update(model)
                running += loss.item()
            eval_model = ema.module if ema is not None else model
            probs, labels = predict(eval_model, val_dl, device, tta=False)
            m = per_class_metrics(probs, labels, classes)
            aucs = [v["auc"] for v in m.values() if "auc" in v]
            mean_auc = float(np.mean(aucs))
            history.append({"epoch": epoch, "loss": running / len(train_dl), "val_mean_auc": mean_auc})
            logger.info("epoch %d/%d loss %.4f val mean AUC %.4f (%.0fs)", epoch, args.epochs, running / len(train_dl), mean_auc, time.time() - t0)
            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save({"model": eval_model.state_dict(), "epoch": epoch, "args": vars(args), "classes": classes}, best_path)

    # Release the training workers before evaluation.
    del train_dl
    import gc
    gc.collect()

    # Evaluate the best checkpoint: thresholds from validation, final numbers on the untouched test split.
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    vp, vl = predict(model, val_dl, device)
    thresholds = tune_thresholds(vp, vl, classes, args.min_specificity)
    # A finding needs at least this much model probability, so near-zero scores never count as "detected".
    thresholds = {c: max(t, args.min_probability) for c, t in thresholds.items()}
    tp_, tl = predict(model, test_dl, device)
    test_metrics = per_class_metrics(tp_, tl, classes, thresholds)
    val_metrics = per_class_metrics(vp, vl, classes, thresholds)

    report = {
        "backbone": args.backbone, "image_size": args.size, "classes": classes,
        "class_names": meta.get("names", {}), "thresholds": thresholds, "excluded_classes": excluded,
        "threshold_rule": (f"lowest threshold with validation specificity >= {args.min_specificity:.2f}, "
                           f"never below probability {args.min_probability:.2f}"),
        "best_epoch": int(torch.load(best_path, map_location="cpu")["epoch"]), "best_val_mean_auc": best_auc,
        "val": val_metrics, "test": test_metrics, "history": history,
        "data": meta.get("summary", {}), "mean": MEAN, "std": STD,
    }
    (OUT_DIR / f"{args.name}_report.json").write_text(json.dumps(report, indent=2))

    # Export to ONNX for the CPU backend, with class activation maps as a second output.
    model.eval().cpu()
    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(CamExportWrapper(model), dummy, OUT_DIR / f"{args.name}.onnx",
                      input_names=["image"], output_names=["logits", "cams"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cams": {0: "batch"}},
                      opset_version=17, dynamo=False)
    logger.info("saved %s.onnx and report; test metrics:", args.name)
    for c in classes:
        t = test_metrics[c]
        logger.info("  %-26s n=%4d pos=%3d AUC=%s sens=%s spec=%s", c, t["n"], t["n_pos"],
                    f"{t['auc']:.3f}" if "auc" in t else "  -  ",
                    f"{t['sensitivity']:.2f}" if t.get("sensitivity") is not None else " - ",
                    f"{t['specificity']:.2f}" if t.get("specificity") is not None else " - ")


if __name__ == "__main__":
    main()
