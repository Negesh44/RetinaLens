"""ONNX Runtime engine for the DR severity model. Imports no PyTorch, so the API server stays small."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort


class ONNXRetinaEngine:
    """Thread-safe, lightweight CPU inference runner using onnxruntime."""

    def __init__(self, onnx_path: Union[str, Path]):
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {self.onnx_path}")

        # Load metadata sidecar
        meta_path = self.onnx_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "backbone_name": "efficientnet_b0",
                "head_type": "regression",
                "image_size": 224,
                "thresholds": [0.5, 1.5, 2.5, 3.5],
                "temperature": 1.0,
            }

        # Initialize onnxruntime session with optimized CPU threads
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Runs forward inference on preprocessed batch or single image [N, 3, H, W]."""
        if input_array.ndim == 3:
            input_array = np.expand_dims(input_array, axis=0)
        input_array = input_array.astype(np.float32)

        raw_pred = self.session.run(None, {self.input_name: input_array})[0]
        return raw_pred
