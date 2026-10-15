@echo off
REM ==============================================================================
REM RetinaLens: Windows Batch Pipeline Runner
REM ==============================================================================
echo ==========================================================================
echo           RetinaLens: End-to-End Pipeline ^& Reproducibility Suite         
echo ==========================================================================

echo.
echo [Step 1/5] Running Model Training, Threshold Optimization ^& ONNX Export...
python -m ml.train --smoke-test
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [Step 2/5] Running Grad-CAM Adebayo Randomization Sanity Check...
python scripts\run_gradcam_sanity_check.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [Step 3/5] Evaluating Cross-Dataset Generalization ^& Robustness Sweeps...
python -m ml.evaluate_generalization
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [Step 4/5] Seeding Demonstration Screening Cohort ^& PDF Reports...
python scripts\seed_demo_data.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [Step 5/5] Executing Pytest Test Suite (ML, API, Quality, Reports)...
python -m pytest tests\
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ==========================================================================
echo           All RetinaLens Experiments ^& Validations Completed!             
echo ==========================================================================
echo Backend:  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
echo Frontend: cd frontend ^&^& npm run dev
echo ==========================================================================
