#!/usr/bin/env bash
# ==============================================================================
# RetinaLens: End-to-End Clinical AI Experimentation Pipeline Runner
# ==============================================================================
set -e

echo "=========================================================================="
echo "          RetinaLens: End-to-End Pipeline & Reproducibility Suite         "
echo "=========================================================================="

# 1. Smoke-test training, threshold optimization, calibration, & ONNX export
echo ""
echo "[Step 1/5] Running Model Training, Threshold Optimization & ONNX Export..."
python -m ml.train --smoke-test

# 2. Grad-CAM Adebayo Sanity Check
echo ""
echo "[Step 2/5] Running Grad-CAM Adebayo Randomization Sanity Check..."
python scripts/run_gradcam_sanity_check.py

# 3. Cross-dataset generalization & controlled degradation sweeps
echo ""
echo "[Step 3/5] Evaluating Cross-Dataset Generalization & Robustness Sweeps..."
python -m ml.evaluate_generalization

# 4. Populate Demonstration Database & PDF Reports
echo ""
echo "[Step 4/5] Seeding Demonstration Screening Cohort & PDF Reports..."
python scripts/seed_demo_data.py

# 5. Run Pytest Test Suite
echo ""
echo "[Step 5/5] Executing Pytest Test Suite (ML, API, Quality, Reports)..."
python -m pytest tests/

echo ""
echo "=========================================================================="
echo "          All RetinaLens Experiments & Validations Completed!             "
echo "=========================================================================="
echo "Artifacts generated:"
echo " - Model Weights:   checkpoints/efficientnet_b0_regression_best.pth"
echo " - ONNX CPU Model:  checkpoints/efficientnet_b0_retinalens.onnx"
echo " - Metrics JSON:    runs/efficientnet_b0_metrics.json"
echo " - Robustness MD:   results/ROBUSTNESS_FINDINGS.md"
echo " - Charts:          generated_charts/"
echo ""
echo "To launch the full system locally:"
echo " Backend:  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
echo " Frontend: cd frontend && npm run dev"
echo "=========================================================================="
