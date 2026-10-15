"""Unit tests for Multi-Disease Retinal Deep Learning Models and Adaptation."""

import pytest
import torch
import numpy as np
from ml.models import RetinaModel, create_retina_model, DEFAULT_DISEASES
from ml.loss import MultiTaskRetinaLoss, calculate_multi_disease_metrics
from ml.pathologies import DISEASE_CATALOG, PATHOLOGY_BIOMARKER_METADATA


def test_disease_catalog_completeness():
    """Validates that all essential clinical diseases and biomarkers are cataloged."""
    expected_diseases = [
        "diabetic_retinopathy",
        "amd",
        "glaucoma",
        "retinal_detachment",
        "macular_hole_pucker",
        "vascular_occlusion",
        "csc",
        "inherited_dystrophy",
        "ocular_cancer",
        "hypertensive_retinopathy",
        "pathologic_myopia",
        "diabetic_macular_edema",
        "papilledema",
        "chorioretinitis",
        "normal",
    ]
    for d_id in expected_diseases:
        assert d_id in DISEASE_CATALOG, f"Missing disease: {d_id}"
        d_def = DISEASE_CATALOG[d_id]
        assert d_def.name
        assert d_def.summary
        assert len(d_def.hallmark_findings) > 0


def test_retina_model_regression_mode():
    """Tests standard backward-compatible DR regression head."""
    model = create_retina_model(backbone_name="efficientnet_b0", pretrained=False, head_type="regression")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2,)


def test_retina_model_multitask_mode():
    """Tests dual-head multi-task model (DR grade + 14 retinal disease logits)."""
    model = create_retina_model(
        backbone_name="efficientnet_b0",
        pretrained=False,
        head_type="multitask",
        num_diseases=len(DEFAULT_DISEASES),
    )
    x = torch.randn(2, 3, 224, 224)
    res = model(x, return_dict=True)

    assert isinstance(res, dict)
    assert "dr_grade" in res
    assert "disease_logits" in res
    assert "disease_probs" in res
    assert res["dr_grade"].shape == (2,)
    assert res["disease_logits"].shape == (2, len(DEFAULT_DISEASES))
    assert res["disease_probs"].shape == (2, len(DEFAULT_DISEASES))

    # Test predict_multi_disease
    pred_dict = model.predict_multi_disease(x[:1])
    assert len(pred_dict) == len(DEFAULT_DISEASES)
    for k in DEFAULT_DISEASES:
        assert k in pred_dict
        assert 0.0 <= pred_dict[k] <= 1.0


def test_multitask_retina_loss():
    """Tests multi-task loss computation."""
    criterion = MultiTaskRetinaLoss(dr_weight=1.0, disease_weight=1.5)
    dr_preds = torch.tensor([1.2, 3.1])
    dr_targets = torch.tensor([1.0, 3.0])
    disease_logits = torch.randn(2, len(DEFAULT_DISEASES))
    disease_targets = torch.zeros(2, len(DEFAULT_DISEASES))
    disease_targets[0, 0] = 1.0  # DR positive
    disease_targets[1, 2] = 1.0  # Glaucoma positive

    loss_dict = criterion(dr_preds, dr_targets, disease_logits, disease_targets)
    assert "loss" in loss_dict
    assert "dr_loss" in loss_dict
    assert "disease_loss" in loss_dict
    assert loss_dict["loss"].item() > 0.0


def test_multi_disease_metrics():
    """Tests multi-disease clinical metrics calculation."""
    y_true = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    y_probs = np.array([
        [0.9, 0.1, 0.1, 0.05],
        [0.2, 0.85, 0.1, 0.1],
        [0.8, 0.75, 0.2, 0.05],
        [0.1, 0.15, 0.9, 0.05],
    ])
    labels = ["dr", "amd", "glaucoma", "rd"]
    metrics = calculate_multi_disease_metrics(y_true, y_probs, labels, threshold=0.5)

    assert "mean_auc" in metrics
    assert "mean_f1" in metrics
    assert "per_disease" in metrics
    assert len(metrics["per_disease"]) == 4
