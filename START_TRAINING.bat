@echo off
REM Quick Training Launcher for Cloud Removal Model
REM This script ensures you're in the correct directory

echo ================================================================================
echo   CLOUD REMOVAL MODEL - TRAINING LAUNCHER
echo ================================================================================
echo.

REM Navigate to correct directory
cd /d "c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"

echo Current directory: %CD%
echo.

echo ================================================================================
echo   STEP 1: PREPARE DATASET
echo ================================================================================
echo.

python prepare_dataset.py --interactive

echo.
echo ================================================================================
echo   STEP 2: TRAIN MODEL
echo ================================================================================
echo.

python quick_train.py

echo.
echo ================================================================================
echo   TRAINING COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo   1. Copy trained model: copy checkpoints\gen_epoch_20.pth gen_epoch_5.pth
echo   2. Run app: python -m streamlit run src/app.py
echo.

pause
