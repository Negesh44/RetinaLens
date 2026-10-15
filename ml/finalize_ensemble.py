"""Combine trained multi-disease models into a calibrated, rule-checked ensemble.

Everything is chosen on the validation split; the test split is used once, at the end, to measure.

1. Predict validation and test logits for every model (with flip test-time augmentation; cached).
2. Pick the subset of models whose averaged logits give the best validation mean AUC.
3. Fit per-condition Platt calibration on validation, so a displayed 90% means about 90%.
4. Tune decision thresholds (validation specificity >= 90%, probability floor 0.20).
5. Tune a consistency rule: a specific condition is reported only if the overall
   "any abnormality" output agrees (p_abnormal >= gate), unless its own probability >= override.
6. Measure the final system, and the previous single model, on the untouched test split.

Usage (GPU environment):
    .venv-train/Scripts/python.exe -m ml.finalize_ensemble retina_multidisease retina_md_b3_ema retina_md_v2s_ema
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ml.train_multidisease import DATA_DIR, MEAN, OUT_DIR, STD, FundusDataset, build_transforms, predict, tune_thresholds

MIN_SPECIFICITY = 0.90
MIN_PROBABILITY = 0.20


def model_logits(name, df_val, df_test, classes, device):
    cache = OUT_DIR / f"{name}_logits.npz"
    if cache.exists():
        z = np.load(cache)
        return z["val"], z["test"]
    ckpt = torch.load(OUT_DIR / f"{name}_best.pt", map_location=device)
    a = ckpt["args"]
    assert ckpt["classes"] == classes, f"{name} was trained on different classes"
    model = timm.create_model(a["backbone"], pretrained=False, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    out = []
    for d in (df_val, df_test):
        dl = DataLoader(FundusDataset(d, classes, build_transforms(a["size"], False)), batch_size=24, num_workers=6)
        p, _ = predict(model, dl, device, tta=True)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out.append(np.log(p / (1 - p)))
    np.savez(cache, val=out[0], test=out[1])
    return out[0], out[1]


def mean_auc(probs, labels, classes):
    aucs = []
    for j in range(len(classes)):
        m = labels[:, j] >= 0
        y = labels[m, j]
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, probs[m, j]))
    return float(np.mean(aucs))


def fit_platt(z, labels, classes):
    params = {}
    for j, c in enumerate(classes):
        m = labels[:, j] >= 0
        lr = LogisticRegression(C=100.0, max_iter=1000)
        lr.fit(z[m, j:j + 1], labels[m, j].astype(int))
        params[c] = [float(lr.coef_[0, 0]), float(lr.intercept_[0])]
    return params


def calibrate(z, params, classes):
    a = np.array([params[c][0] for c in classes])
    b = np.array([params[c][1] for c in classes])
    return 1.0 / (1.0 + np.exp(-(z * a + b)))


def decide(p, classes, thresholds, gate, override):
    """Boolean detection matrix for the disease classes (column order = classes, abnormal column included)."""
    ia = classes.index("abnormal")
    th = np.array([thresholds[c] for c in classes])
    det = p >= th
    agree = (p[:, ia:ia + 1] >= gate) | (p >= override)
    det = det & agree
    det[:, ia] = p[:, ia] >= thresholds["abnormal"]
    return det


def evaluate(det, p, labels, classes):
    ia = classes.index("abnormal")
    per_class = {}
    bal = []
    for j, c in enumerate(classes):
        m = labels[:, j] >= 0
        y, d = labels[m, j] == 1, det[m, j]
        tp, fn = int((d & y).sum()), int((~d & y).sum())
        tn, fp = int((~d & ~y).sum()), int((d & ~y).sum())
        sens = tp / (tp + fn) if tp + fn else None
        spec = tn / (tn + fp) if tn + fp else None
        entry = {"n": int(m.sum()), "n_pos": int(y.sum()), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                 "sensitivity": sens, "specificity": spec, "accuracy": (tp + tn) / max(1, m.sum())}
        if 0 < y.sum() < len(y):
            entry["auc"] = float(roc_auc_score(y, p[m, j]))
        per_class[c] = entry
        if c != "abnormal" and sens is not None and spec is not None:
            bal.append((sens + spec) / 2)
    # Overall verdict as the app shows it: abnormal if any condition or the abnormality output fires.
    verdict = det.any(axis=1)
    truth = labels[:, ia] == 1
    healthy = ~truth
    verdict_sens = float((verdict & truth).sum() / max(1, truth.sum()))
    verdict_spec = float((~verdict & healthy).sum() / max(1, healthy.sum()))
    disease_cols = [j for j, c in enumerate(classes) if c != "abnormal"]
    return {
        "per_class": per_class,
        "macro_balanced_accuracy": float(np.mean(bal)),
        "verdict": {
            "sensitivity": verdict_sens,
            "specificity": verdict_spec,
            "balanced_accuracy": (verdict_sens + verdict_spec) / 2,
            "healthy_eyes": int(healthy.sum()),
            # Share of truly healthy eyes for which no specific condition was reported at all.
            "healthy_with_no_condition_reported": float((~det[healthy][:, disease_cols].any(axis=1)).mean()),
        },
    }


def main():
    names = sys.argv[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    base = json.loads((OUT_DIR / f"{names[0]}_report.json").read_text())
    classes = base["classes"]
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    yv, yt = va[classes].to_numpy(np.float32), te[classes].to_numpy(np.float32)

    logits = {n: model_logits(n, va, te, classes, device) for n in names}
    single_val_auc = {n: mean_auc(1 / (1 + np.exp(-logits[n][0])), yv, classes) for n in names}
    print("validation mean AUC per model:", {k: round(v, 4) for k, v in single_val_auc.items()})

    # 2. Member selection on validation.
    best_subset, best_auc = None, -1.0
    for r in range(1, len(names) + 1):
        for subset in itertools.combinations(names, r):
            zv = np.mean([logits[n][0] for n in subset], axis=0)
            auc = mean_auc(1 / (1 + np.exp(-zv)), yv, classes)
            print(f"  subset {subset}: val mean AUC {auc:.4f}")
            if auc > best_auc + 1e-4:
                best_subset, best_auc = subset, auc
    print("chosen members:", best_subset, round(best_auc, 4))
    zv = np.mean([logits[n][0] for n in best_subset], axis=0)
    zt = np.mean([logits[n][1] for n in best_subset], axis=0)

    # 3-4. Calibration and thresholds on validation.
    platt = fit_platt(zv, yv, classes)
    pv, pt = calibrate(zv, platt, classes), calibrate(zt, platt, classes)
    thresholds = tune_thresholds(pv, yv, classes, MIN_SPECIFICITY)
    thresholds = {c: max(t, MIN_PROBABILITY) for c, t in thresholds.items()}

    # 5. Consistency rule on validation (gate 0 = rule off).
    def objective(ev):
        return (ev["macro_balanced_accuracy"] + ev["verdict"]["balanced_accuracy"]) / 2

    best = (objective(evaluate(decide(pv, classes, thresholds, 0.0, 2.0), pv, yv, classes)), 0.0, 2.0)
    for gate in np.arange(0.20, 0.96, 0.05):
        for override in (0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 2.0):
            score = objective(evaluate(decide(pv, classes, thresholds, gate, override), pv, yv, classes))
            if score > best[0] + 1e-4:
                best = (score, float(gate), float(override))
    _, gate, override = best
    print(f"consistency rule: gate={gate:.2f} override={override:.2f} (val objective {best[0]:.4f})")

    # 6. Test measurement: final system and the previous single model for comparison.
    final_val = evaluate(decide(pv, classes, thresholds, gate, override), pv, yv, classes)
    final_test = evaluate(decide(pt, classes, thresholds, gate, override), pt, yt, classes)
    old = json.loads((OUT_DIR / "retina_multidisease_report.json").read_text())
    old_p = 1 / (1 + np.exp(-logits["retina_multidisease"][1])) if "retina_multidisease" in logits else None
    baseline = None
    if old_p is not None:
        baseline = evaluate(decide(old_p, classes, old["thresholds"], 0.0, 2.0), old_p, yt, classes)

    members = []
    for n in best_subset:
        ck = torch.load(OUT_DIR / f"{n}_best.pt", map_location="cpu")
        members.append({"name": n, "onnx": f"{n}.onnx", "backbone": ck["args"]["backbone"],
                        "image_size": ck["args"]["size"], "best_epoch": int(ck["epoch"])})
    report = {
        "type": "ensemble",
        "members": members,
        "classes": classes,
        "class_names": base.get("class_names", {}),
        "excluded_classes": base.get("excluded_classes", []),
        "calibration": platt,
        "thresholds": thresholds,
        "consistency_rule": {"gate": gate, "override": override,
                             "description": "a condition is reported only if p(any abnormality) >= gate, "
                                            "unless its own probability >= override"},
        "threshold_rule": f"validation specificity >= {MIN_SPECIFICITY:.2f}, probability floor {MIN_PROBABILITY:.2f}",
        "mean": MEAN, "std": STD,
        "val_mean_auc_per_model": single_val_auc,
        "val_mean_auc": best_auc,
        "val": final_val,
        "test": final_test["per_class"],
        "test_summary": {k: v for k, v in final_test.items() if k != "per_class"},
        "baseline_single_model_test": baseline,
        "data": base.get("data", {}),
    }
    (OUT_DIR / "retina_ensemble.json").write_text(json.dumps(report, indent=2))

    def line(ev, c):
        e = ev["per_class"][c]
        f = lambda v: f"{v:.2f}" if v is not None else " - "
        return f"AUC {e.get('auc', float('nan')):.3f} sens {f(e['sensitivity'])} spec {f(e['specificity'])}"

    print("\nTEST (final ensemble)                               | previous single model")
    for c in classes:
        print(f"  {c:26} n_pos={final_test['per_class'][c]['n_pos']:4}  {line(final_test, c)}  |  {line(baseline, c) if baseline else ''}")
    for label, ev in (("final", final_test), ("previous", baseline)):
        if ev:
            v = ev["verdict"]
            print(f"  {label:8} verdict: sens {v['sensitivity']:.3f} spec {v['specificity']:.3f} | healthy eyes with no condition reported: "
                  f"{v['healthy_with_no_condition_reported']:.3f} of {v['healthy_eyes']} | macro balanced acc {ev['macro_balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
