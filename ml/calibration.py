"""Probability Calibration via Temperature Scaling and Reliability Diagrams.

Implements:
1. Temperature Scaling optimization on validation set.
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
3. Reliability Diagram generation comparing uncalibrated vs calibrated probabilities.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Calibrates predictions using Post-Hoc Temperature Scaling (Guo et al., 2017)."""

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1 / T."""
        temp = self.temperature.clamp(min=0.05, max=10.0)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Optimizes temperature T on validation logits to minimize NLL."""
        self.temperature.data.fill_(1.0)
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = nll_criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        final_temp = float(self.temperature.item())
        return final_temp


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: Array of predicted probabilities [N, num_classes].
        labels: Ground truth class indices [N].
        n_bins: Number of equal-width confidence bins between 0 and 1.

    Returns:
        Dictionary containing ECE, MCE, and per-bin statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    total_samples = len(labels)
    if total_samples == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_count / total_samples) * bin_error
            mce = max(mce, bin_error)

            bin_stats.append({
                "bin_idx": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": int(bin_count),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_error, 4),
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
                "error": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bins": bin_stats,
    }


def plot_reliability_diagram(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    save_path: Union[str, Path],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Generates and saves a publication-ready reliability diagram comparison.

    Returns dict with ECE before and after.
    """
    uncal_res = compute_ece(uncalibrated_probs, labels, n_bins=n_bins)
    cal_res = compute_ece(calibrated_probs, labels, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, res, title, color in [
        (axes[0], uncal_res, f"Uncalibrated (ECE: {uncal_res['ece']:.3f})", "#ef4444"),
        (axes[1], cal_res, f"Calibrated (ECE: {cal_res['ece']:.3f})", "#10b981"),
    ]:
        confs = [b["confidence"] for b in res["bins"] if b["count"] > 0]
        accs = [b["accuracy"] for b in res["bins"] if b["count"] > 0]

        # Ideal identity line
        ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect Calibration")
        ax.bar(
            confs,
            accs,
            width=0.07,
            alpha=0.6,
            edgecolor=color,
            color=color,
            label="Observed Accuracy",
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Empirical Accuracy", fontsize=11)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    return {
        "ece_before": uncal_res["ece"],
        "ece_after": cal_res["ece"],
        "mce_before": uncal_res["mce"],
        "mce_after": cal_res["mce"],
    }
