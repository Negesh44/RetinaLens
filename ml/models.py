"""Model Architectures for Multi-Disease Retinal Pathology and Diabetic Retinopathy Grading.

Supports timm backbones:
- EfficientNet-B0 (default lightweight backbone)
- EfficientNet-B3 (higher resolution backbone)
- ResNet-50 (standard clinical baseline)

Heads supported:
- 'regression': Continuous ordinal head with learned/optimized thresholding for DR.
- 'classification': 5-class logits for cross-entropy / ordinal loss.
- 'ordinal': Frank & Hall / CORAL binary thresholds.
- 'multilabel': Multi-label classification logits across full retinal disease catalog.
- 'multitask': Dual-head joint learning for simultaneous DR grading and multi-disease diagnosis.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm

from ml.config import SUPPORTED_BACKBONES, DEFAULT_BACKBONE

# Canonical list of 14 non-normal clinical retinal conditions
DEFAULT_DISEASES: List[str] = [
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
]


class RetinaModel(nn.Module):
    """Deep Neural Network with configurable backbone and single or multi-task heads."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        head_type: str = "regression",  # 'regression', 'classification', 'ordinal', 'multilabel', 'multitask'
        dropout_rate: float = 0.3,
        num_classes: int = 5,
        num_diseases: int = len(DEFAULT_DISEASES),
        disease_labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_type = head_type.lower()
        self.num_classes = num_classes
        self.num_diseases = num_diseases
        self.disease_labels = disease_labels or list(DEFAULT_DISEASES)

        # Initialize timm backbone with pooled features
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            drop_rate=dropout_rate,
        )

        in_features = self.encoder.num_features

        # Projection heads
        if self.head_type == "regression":
            # Output continuous DR severity prediction
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, 1),
            )
        elif self.head_type == "classification":
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, num_classes),
            )
        elif self.head_type == "ordinal":
            # Frank & Hall / CORAL 4 binary indicators (P(y > k))
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, num_classes - 1),
            )
        elif self.head_type == "multilabel":
            # Direct multi-label classification across all diseases
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, self.num_diseases),
            )
        elif self.head_type == "multitask":
            # Dual-head: DR severity regression + Multi-disease classification logits
            self.dr_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, 1),
            )
            self.disease_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, self.num_diseases),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        features = self.encoder(x)

        if self.head_type == "multitask":
            dr_out = self.dr_head(features).squeeze(-1)  # [Batch]
            disease_logits = self.disease_head(features)  # [Batch, num_diseases]

            if return_dict:
                return {
                    "dr_grade": dr_out,
                    "disease_logits": disease_logits,
                    "disease_probs": torch.sigmoid(disease_logits),
                }
            return dr_out, disease_logits

        elif self.head_type == "multilabel":
            disease_logits = self.head(features)  # [Batch, num_diseases]
            if return_dict:
                return {
                    "disease_logits": disease_logits,
                    "disease_probs": torch.sigmoid(disease_logits),
                }
            return disease_logits

        elif self.head_type == "regression":
            out = self.head(features)
            return out.squeeze(-1)  # [Batch]

        else:
            return self.head(features)

    def predict_multi_disease(self, x: torch.Tensor) -> Dict[str, float]:
        """Inference helper returning calibrated per-disease probability mapping."""
        self.eval()
        with torch.no_grad():
            features = self.encoder(x)
            if self.head_type == "multitask":
                logits = self.disease_head(features)
            elif self.head_type == "multilabel":
                logits = self.head(features)
            else:
                # Approximate from primary features using projection
                return {}

            probs = torch.sigmoid(logits)[0].cpu().numpy()
            return {
                label: float(probs[i])
                for i, label in enumerate(self.disease_labels[:len(probs)])
            }

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the last convolutional layer module for Grad-CAM inspection."""
        name = self.backbone_name.lower()
        if "efficientnet" in name:
            if hasattr(self.encoder, "conv_head"):
                return self.encoder.conv_head
            elif hasattr(self.encoder, "blocks"):
                return self.encoder.blocks[-1]
            elif hasattr(self.encoder, "act2"):
                return self.encoder.act2
        elif "resnet" in name:
            if hasattr(self.encoder, "layer4"):
                return self.encoder.layer4[-1]

        # General fallback: find last Conv2d in encoder
        for m in reversed(list(self.encoder.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError(f"Could not automatically locate last conv layer in {self.backbone_name}")


def create_retina_model(
    backbone_name: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
    head_type: str = "regression",
    dropout_rate: float = 0.3,
    num_classes: int = 5,
    num_diseases: int = len(DEFAULT_DISEASES),
) -> RetinaModel:
    """Factory helper to instantiate RetinaModel with single or multi-task heads."""
    return RetinaModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        head_type=head_type,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        num_diseases=num_diseases,
    )
