"""End-to-End Training, Validation, and Calibration Pipeline for RetinaLens.

Usage:
    python -m ml.train --config ml/configs/default_b0.yaml
    python -m ml.train --smoke-test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import (
    RANDOM_SEED,
    DATA_DIR,
    CHECKPOINTS_DIR,
    CHARTS_DIR,
    RUNS_DIR,
    DEFAULT_BACKBONE,
    BACKBONE_IMAGE_SIZES,
)
from ml.data import (
    load_dataset_metadata,
    generate_synthetic_dataset,
    prepare_stratified_splits,
    get_class_distribution_report,
    create_weighted_sampler,
    RetinaFundusDataset,
)
from ml.preprocessing import get_train_transforms, get_val_transforms
from ml.models import RetinaModel, create_retina_model
from ml.loss import (
    OrdinalLoss,
    ThresholdOptimizer,
    continuous_to_probabilities,
    calculate_clinical_metrics,
)
from ml.calibration import TemperatureScaler, plot_reliability_diagram
from ml.export import export_retina_model_to_onnx

logger = logging.getLogger("retinalens.train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds across random, numpy, and torch for strict reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_type: str = "regression",
) -> float:
    """Trains model for one epoch and returns average training loss."""
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if head_type == "regression":
            loss = criterion(outputs, targets.float())
        elif head_type == "ordinal":
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold_optimizer: Optional[ThresholdOptimizer] = None,
    head_type: str = "regression",
    fit_thresholds: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Evaluates model on validation/test set.

    Returns:
        (val_loss, y_true, y_pred, probs, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        if head_type == "regression":
            loss = criterion(outputs, targets.float())
        elif head_type == "ordinal":
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)

        total_loss += loss.item() * len(targets)
        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

    val_loss = total_loss / len(loader.dataset)
    raw_outputs = torch.cat(all_outputs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    if head_type == "regression":
        if fit_thresholds and threshold_optimizer is not None:
            threshold_optimizer.fit(raw_outputs, y_true)

        thresholds = threshold_optimizer.thresholds if threshold_optimizer else [0.5, 1.5, 2.5, 3.5]
        y_pred = threshold_optimizer.discretize(raw_outputs, thresholds) if threshold_optimizer else np.clip(np.round(raw_outputs), 0, 4).astype(int)
        probs = continuous_to_probabilities(raw_outputs)
    elif head_type == "ordinal":
        # Multi-binary sigmoids
        sigmoids = 1.0 / (1.0 + np.exp(-raw_outputs))
        y_pred = np.sum(sigmoids > 0.5, axis=1)
        # Construct cumulative class probabilities
        p_greater = sigmoids
        probs = np.zeros((len(y_pred), 5), dtype=float)
        probs[:, 0] = 1.0 - p_greater[:, 0]
        probs[:, 1] = np.clip(p_greater[:, 0] - p_greater[:, 1], 0, 1)
        probs[:, 2] = np.clip(p_greater[:, 1] - p_greater[:, 2], 0, 1)
        probs[:, 3] = np.clip(p_greater[:, 2] - p_greater[:, 3], 0, 1)
        probs[:, 4] = p_greater[:, 3]
        probs = probs / np.sum(probs, axis=1, keepdims=True)
    else:
        # Standard classification
        exp_logits = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        y_pred = np.argmax(probs, axis=1)

    metrics = calculate_clinical_metrics(y_true, y_pred, probs=probs)
    return val_loss, y_true, y_pred, probs, metrics


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
):
    """Plots and saves loss and QWK curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#0284c7")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ef4444")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # QWK curve
    axes[1].plot(epochs, history["val_qwk"], "^-", label="Validation QWK", color="#10b981")
    axes[1].set_title("Quadratic Weighted Kappa (QWK)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Accuracy curve
    axes[2].plot(epochs, history["val_acc"], "d-", label="Validation Accuracy", color="#8b5cf6")
    axes[2].set_title("Validation Accuracy", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_training_pipeline(
    config_dict: Dict,
    smoke_test: bool = False,
) -> Dict[str, Union[float, str, Dict]]:
    """Executes the full training, calibration, evaluation, and ONNX export pipeline."""
    set_seed(RANDOM_SEED)

    backbone_name = config_dict.get("backbone", DEFAULT_BACKBONE)
    head_type = config_dict.get("head_type", "regression")
    image_size = config_dict.get("image_size", BACKBONE_IMAGE_SIZES.get(backbone_name, 224))
    epochs = 1 if smoke_test else config_dict.get("epochs", 15)
    batch_size = 4 if smoke_test else config_dict.get("batch_size", 16)
    lr = float(config_dict.get("lr", 3e-4))
    use_ben_graham = config_dict.get("use_ben_graham", True)
    dataset_name = config_dict.get("dataset", "aptos")

    device = torch.device("cuda" if torch.cuda.is_available() and not smoke_test else "cpu")
    logger.info("Starting pipeline on device: %s (smoke_test=%s)", device, smoke_test)

    # 1. Dataset Loading
    df = None
    if not smoke_test:
        df = load_dataset_metadata(dataset_name)

    if df is None or len(df) == 0:
        logger.info("Using synthetic dataset generator for training (dataset: %s)", dataset_name)
        num_synth = 24 if smoke_test else 120
        synth_dir = DATA_DIR / ("smoke_test" if smoke_test else "synthetic")
        df = generate_synthetic_dataset(synth_dir, num_samples=num_synth)

    dist_report = get_class_distribution_report(df)
    logger.info("Dataset distribution: %s", json.dumps(dist_report["percentages"]))

    train_df, val_df = prepare_stratified_splits(df, val_ratio=0.25, seed=RANDOM_SEED)
    logger.info("Train set: %d samples, Val set: %d samples", len(train_df), len(val_df))

    # Datasets and Loaders
    train_dataset = RetinaFundusDataset(
        train_df["image_path"].tolist(),
        train_df["grade"].tolist(),
        target_size=image_size,
        use_ben_graham=use_ben_graham,
        transforms=get_train_transforms(image_size),
    )
    val_dataset = RetinaFundusDataset(
        val_df["image_path"].tolist(),
        val_df["grade"].tolist(),
        target_size=image_size,
        use_ben_graham=use_ben_graham,
        transforms=get_val_transforms(image_size),
    )

    sampler = create_weighted_sampler(train_df["grade"].tolist())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model, Loss, Optimizer
    model = create_retina_model(
        backbone_name=backbone_name,
        pretrained=not smoke_test,
        head_type=head_type,
    ).to(device)

    if head_type == "regression":
        criterion = nn.SmoothL1Loss()
    elif head_type == "ordinal":
        criterion = OrdinalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    threshold_optimizer = ThresholdOptimizer() if head_type == "regression" else None

    # Tracking
    history = {"train_loss": [], "val_loss": [], "val_qwk": [], "val_acc": []}
    best_qwk = -1.0
    best_weights_path = CHECKPOINTS_DIR / f"{backbone_name}_{head_type}_best.pth"
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, head_type)
        val_loss, y_true, y_pred, probs, metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            threshold_optimizer=threshold_optimizer,
            head_type=head_type,
            fit_thresholds=True,
        )
        scheduler.step()

        qwk = metrics["quadratic_weighted_kappa"]
        acc = metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(qwk)
        history["val_acc"].append(acc)

        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f | Val QWK: %.4f | Val Acc: %.4f",
            epoch, epochs, train_loss, val_loss, qwk, acc,
        )

        if qwk > best_qwk or epoch == 1:
            best_qwk = qwk
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone_name": backbone_name,
                "head_type": head_type,
                "image_size": image_size,
                "thresholds": threshold_optimizer.thresholds if threshold_optimizer else None,
                "metrics": metrics,
            }, str(best_weights_path))

    # 4. Final Evaluation & Calibration on Best Model
    checkpoint = torch.load(str(best_weights_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, y_true, y_pred, uncal_probs, final_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        threshold_optimizer=threshold_optimizer,
        head_type=head_type,
        fit_thresholds=False,
    )

    # Convert validation probabilities into pseudo-logits for Temperature Scaling
    val_pseudo_logits = torch.tensor(
        np.log(np.clip(uncal_probs, 1e-7, 1.0 - 1e-7)),
        dtype=torch.float32,
    )
    val_labels_tensor = torch.tensor(y_true, dtype=torch.long)

    temp_scaler = TemperatureScaler()
    optimal_temp = temp_scaler.fit(val_pseudo_logits, val_labels_tensor)
    calibrated_logits = temp_scaler(val_pseudo_logits).detach().numpy()
    cal_exp = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
    calibrated_probs = cal_exp / np.sum(cal_exp, axis=1, keepdims=True)

    # Plot Reliability Diagram
    diagram_path = CHARTS_DIR / f"{backbone_name}_calibration_diagram.png"
    calib_summary = plot_reliability_diagram(
        uncalibrated_probs=uncal_probs,
        calibrated_probs=calibrated_probs,
        labels=y_true,
        save_path=diagram_path,
    )

    # Plot Training Curves
    curves_path = CHARTS_DIR / f"{backbone_name}_training_curves.png"
    plot_training_curves(history, curves_path)

    # 5. Export Model to ONNX
    onnx_save_path = CHECKPOINTS_DIR / f"{backbone_name}_retinalens.onnx"
    export_retina_model_to_onnx(
        model=model,
        save_path=onnx_save_path,
        image_size=image_size,
        thresholds=threshold_optimizer.thresholds if threshold_optimizer else None,
        temperature=optimal_temp,
        verify_equivalence=True,
    )

    # 6. Save Complete Metrics JSON
    results_json_path = RUNS_DIR / f"{backbone_name}_metrics.json"
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_results = {
        "dataset": dataset_name,
        "backbone": backbone_name,
        "head_type": head_type,
        "image_size": image_size,
        "optimal_temperature": round(optimal_temp, 4),
        "calibration": calib_summary,
        "best_validation_metrics": final_metrics,
        "thresholds": threshold_optimizer.thresholds if threshold_optimizer else None,
        "class_distribution": dist_report,
        "charts": {
            "training_curves": str(curves_path.resolve()),
            "calibration_diagram": str(diagram_path.resolve()),
        },
        "artifacts": {
            "checkpoint": str(best_weights_path.resolve()),
            "onnx_model": str(onnx_save_path.resolve()),
        },
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info("Training pipeline completed! Summary saved to %s", results_json_path)
    return summary_results


def main():
    parser = argparse.ArgumentParser(description="RetinaLens Model Training and Evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch CPU verification test")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if args.backbone:
        config["backbone"] = args.backbone
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr

    run_training_pipeline(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
