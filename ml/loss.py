"""Loss Functions, Metric Calculations, and Threshold Optimization.

Implements:
1. Smooth L1 & MSE Regression Losses for ordinal severity.
2. Frank & Hall Ordinal Cross-Entropy.
3. Class-weighted Cross-Entropy.
4. Threshold Optimizer: Optimizes continuous boundaries [th0, th1, th2, th3]
   to strictly maximize Quadratic Weighted Kappa (QWK).
5. Comprehensive metrics suite:
   - Quadratic Weighted Kappa (QWK)
   - Multi-class Accuracy & Macro F1
   - Per-class sensitivity / recall
   - Referable DR (grade >= 2) Sensitivity, Specificity, and ROC-AUC
   - Confusion Matrix
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

from ml.config import REFERABLE_GRADE_THRESHOLD, ICDR_GRADES


class OrdinalLoss(nn.Module):
    """Frank & Hall / CORAL Ordinal Classification Loss using multi-binary cross entropy."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: [batch_size, num_classes - 1]
        # targets shape: [batch_size] with integer grades 0..num_classes-1
        batch_size = targets.size(0)
        num_cuts = self.num_classes - 1
        binary_targets = torch.zeros((batch_size, num_cuts), device=targets.device, dtype=torch.float32)

        for k in range(num_cuts):
            binary_targets[:, k] = (targets > k).float()

        return self.bce(logits, binary_targets)


class ThresholdOptimizer:
    """Optimizes continuous regression thresholds to maximize Quadratic Weighted Kappa."""

    def __init__(self, initial_thresholds: Optional[List[float]] = None):
        if initial_thresholds is None:
            self.thresholds = [0.5, 1.5, 2.5, 3.5]
        else:
            self.thresholds = sorted(initial_thresholds)

    def discretize(self, continuous_preds: np.ndarray, thresholds: Optional[List[float]] = None) -> np.ndarray:
        """Maps continuous floats to discrete grades 0..4 using thresholds."""
        th = sorted(thresholds if thresholds is not None else self.thresholds)
        discrete = np.zeros_like(continuous_preds, dtype=int)
        discrete[continuous_preds >= th[0]] = 1
        discrete[continuous_preds >= th[1]] = 2
        discrete[continuous_preds >= th[2]] = 3
        discrete[continuous_preds >= th[3]] = 4
        return discrete

    def _loss(self, thresholds: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> float:
        # Penalize non-monotonic thresholds
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            return 10.0
        y_pred = self.discretize(preds, list(thresholds))
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if np.isnan(kappa):
            return 0.0
        return -kappa  # Minimize negative kappa

    def fit(self, continuous_preds: np.ndarray, y_true: np.ndarray) -> List[float]:
        """Runs Powell optimization on validation continuous predictions."""
        init_th = np.array(self.thresholds, dtype=float)
        res = minimize(
            self._loss,
            init_th,
            args=(continuous_preds, y_true),
            method="Powell",
            options={"maxiter": 200, "ftol": 1e-4},
        )
        if res.success and (res.x[0] < res.x[1] < res.x[2] < res.x[3]):
            self.thresholds = [round(float(t), 4) for t in res.x]
        else:
            # Fallback if optimization diverged
            self.thresholds = [0.5, 1.5, 2.5, 3.5]
        return self.thresholds


def continuous_to_probabilities(
    continuous_pred: Union[float, np.ndarray],
    sigma: float = 0.65,
) -> np.ndarray:
    """Converts a continuous severity prediction into calibrated 5-class probabilities

    using a Gaussian distance kernel.
    """
    is_scalar = np.isscalar(continuous_pred)
    preds = np.atleast_1d(continuous_pred)

    class_indices = np.arange(5, dtype=float)  # [0, 1, 2, 3, 4]
    # Distances shape: [N, 5]
    diffs = preds[:, None] - class_indices[None, :]
    logits = -(diffs ** 2) / (2 * (sigma ** 2))
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    if is_scalar:
        return probs[0]
    return probs


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, List, Dict]]:
    """Calculates full clinical evaluation metrics suite for diabetic retinopathy.

    Args:
        y_true: Ground truth integer grades [N]
        y_pred: Predicted integer grades [N]
        probs: Class probabilities [N, 5] (optional, for AUC)

    Returns:
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_samples = len(y_true)

    if n_samples == 0:
        return {"error": "Empty evaluation set"}

    # 1. Quadratic Weighted Kappa (QWK)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    if np.isnan(qwk):
        qwk = 0.0

    # 2. Overall Accuracy and Macro F1
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 3. Per-class Recall (Sensitivity per grade 0..4)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_class_dict = {
        ICDR_GRADES[i]: round(float(per_class_rec[i]), 4)
        for i in range(5)
    }

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    # 5. Referable DR Metrics (Grade >= 2)
    binary_true = (y_true >= REFERABLE_GRADE_THRESHOLD).astype(int)
    binary_pred = (y_pred >= REFERABLE_GRADE_THRESHOLD).astype(int)

    # Sensitivity = TP / (TP + FN)
    tp = np.sum((binary_true == 1) & (binary_pred == 1))
    fn = np.sum((binary_true == 1) & (binary_pred == 0))
    tn = np.sum((binary_true == 0) & (binary_pred == 0))
    fp = np.sum((binary_true == 0) & (binary_pred == 1))

    ref_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    ref_auc = None
    if probs is not None and len(np.unique(binary_true)) > 1:
        # Sum probabilities of grades >= 2
        prob_referable = np.sum(probs[:, REFERABLE_GRADE_THRESHOLD:], axis=1)
        try:
            ref_auc = float(roc_auc_score(binary_true, prob_referable))
        except Exception:
            ref_auc = None

    return {
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class_recall": per_class_dict,
        "referable_dr": {
            "sensitivity": round(ref_sensitivity, 4),
            "specificity": round(ref_specificity, 4),
            "auc": round(ref_auc, 4) if ref_auc is not None else None,
            "threshold": REFERABLE_GRADE_THRESHOLD,
        },
        "confusion_matrix": cm.tolist(),
        "total_samples": n_samples,
    }


class MultiTaskRetinaLoss(nn.Module):
    """Joint Multi-Task Loss combining DR continuous regression with Multi-Disease BCE."""

    def __init__(
        self,
        dr_weight: float = 1.0,
        disease_weight: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dr_weight = dr_weight
        self.disease_weight = disease_weight
        self.dr_loss_fn = nn.SmoothL1Loss(beta=0.5)
        self.disease_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        dr_preds: torch.Tensor,
        dr_targets: torch.Tensor,
        disease_logits: torch.Tensor,
        disease_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_dr = self.dr_loss_fn(dr_preds, dr_targets.float())
        l_disease = self.disease_loss_fn(disease_logits, disease_targets.float())
        total_loss = (self.dr_weight * l_dr) + (self.disease_weight * l_disease)
        return {
            "loss": total_loss,
            "dr_loss": l_dr,
            "disease_loss": l_disease,
        }


def calculate_multi_disease_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    disease_labels: List[str],
    threshold: float = 0.5,
) -> Dict[str, Union[float, Dict]]:
    """Calculates multi-label classification metrics across all retinal pathologies.

    Args:
        y_true: Binary indicator ground truth [N, num_diseases]
        y_probs: Predicted probabilities [N, num_diseases]
        disease_labels: List of disease identifier strings
        threshold: Decision threshold for binary classification
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    per_disease = {}
    auc_scores = []

    for i, disease in enumerate(disease_labels):
        if i >= y_true.shape[1] or i >= y_probs.shape[1]:
            continue
        t_col = y_true[:, i]
        p_col = y_probs[:, i]
        b_col = y_pred[:, i]

        tp = np.sum((t_col == 1) & (b_col == 1))
        fn = np.sum((t_col == 1) & (b_col == 0))
        tn = np.sum((t_col == 0) & (b_col == 0))
        fp = np.sum((t_col == 0) & (b_col == 1))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(t_col, b_col, zero_division=0))

        disease_auc = None
        if len(np.unique(t_col)) > 1:
            try:
                disease_auc = float(roc_auc_score(t_col, p_col))
                auc_scores.append(disease_auc)
            except Exception:
                disease_auc = None

        per_disease[disease] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "auc": round(disease_auc, 4) if disease_auc is not None else None,
            "prevalence": int(np.sum(t_col)),
        }

    mean_auc = round(float(np.mean(auc_scores)), 4) if auc_scores else None
    mean_f1 = round(float(np.mean([d["f1_score"] for d in per_disease.values()])), 4) if per_disease else 0.0

    return {
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
        "per_disease": per_disease,
    }

