"""ONNX Export and CPU-Optimized Inference Engine.

Exports PyTorch RetinaModel architectures to ONNX format,
validates graph topology, and verifies numerical equivalence
between PyTorch and onnxruntime.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import numpy as np
import torch
import onnx
import onnxruntime as ort

from ml.config import CHECKPOINTS_DIR, BACKBONE_IMAGE_SIZES
from ml.models import RetinaModel, create_retina_model
from ml.onnx_engine import ONNXRetinaEngine  # noqa: F401  (re-exported for existing imports)

logger = logging.getLogger("retinalens.export")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def export_retina_model_to_onnx(
    model: RetinaModel,
    save_path: Union[str, Path],
    image_size: int = 224,
    thresholds: Optional[List[float]] = None,
    temperature: float = 1.0,
    verify_equivalence: bool = True,
) -> Path:
    """Exports a trained PyTorch model to ONNX with dynamic batch size.

    Args:
        model: Trained RetinaModel.
        save_path: Output .onnx file path.
        image_size: Input spatial resolution.
        thresholds: Discretization thresholds [th0, th1, th2, th3].
        temperature: Calibrated temperature scaling parameter.
        verify_equivalence: If True, tests onnxruntime vs PyTorch output.

    Returns:
        Path to exported ONNX model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    logger.info("Exporting %s to ONNX at %s...", model.backbone_name, save_path)

    input_names = ["input_fundus"]
    output_names = ["prediction"]

    dynamic_axes = {
        "input_fundus": {0: "batch_size"},
        "prediction": {0: "batch_size"},
    }

    # PyTorch ONNX export with TorchScript tracer
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # For older PyTorch without dynamo kwarg
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(save_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph successfully verified by onnx.checker.")

    # Save metadata sidecar JSON
    meta_path = save_path.with_suffix(".json")
    metadata = {
        "backbone_name": model.backbone_name,
        "head_type": model.head_type,
        "image_size": image_size,
        "thresholds": thresholds or [0.5, 1.5, 2.5, 3.5],
        "temperature": temperature,
        "input_names": input_names,
        "output_names": output_names,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Numerical Equivalence Check
    if verify_equivalence:
        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        ort_session = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]

        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        logger.info("ONNX numerical check: max absolute difference = %.6f", max_abs_diff)
        if max_abs_diff > 1e-3:
            logger.warning("Numerical discrepancy between PyTorch and ONNX exceeds 1e-3 (%.6f)", max_abs_diff)
        else:
            logger.info("Numerical equivalence confirmed (diff <= 1e-3).")

    return save_path
