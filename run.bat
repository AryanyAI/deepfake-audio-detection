@echo off
echo ===== DeepFake Audio Detection Pipeline =====
echo This script runs the entire pipeline from data preparation to training
echo.

rem Create necessary directories
if not exist data mkdir data
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist models\checkpoints mkdir models\checkpoints

rem Check if dataset files exist
set DATA_FOUND=0
if exist data\raw\flac_D_ab.tar (
    set DATA_FOUND=1
) else if exist data\raw\ASVspoof_LA_D.zip (
    set DATA_FOUND=1
) else if exist data\raw\LA_D.zip (
    set DATA_FOUND=1
)

if %DATA_FOUND%==0 (
    echo [WARNING] No dataset files found in data\raw directory.
    echo Will create synthetic dataset for testing the pipeline.
    echo.
    echo For better results, download one of these datasets:
    echo - ASVspoof5: https://zenodo.org/records/14498691
    echo - ASVspoof2019: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
    echo.
    echo Press any key to continue with synthetic data or Ctrl+C to cancel...
    pause > nul
)

echo Step 1: Preparing dataset...
python utils\prepare_dataset.py
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Dataset preparation had issues, but continuing with pipeline...
)

echo.
echo Step 2: Preprocessing data...
python utils\preprocess_data.py --subset dev --limit 500 --low_memory --ensure_balanced
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Data preprocessing had issues, but continuing with synthetic data...
)

echo.
echo Step 3: Training model...
python notebooks\aasist_training.py --small_model --epochs 5 --batch_size 16 --max_samples 500 --device cpu --test
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model training failed
    exit /b 1
)

echo.
echo ===== Pipeline completed successfully! =====
echo.
echo Next steps:
echo 1. Test the model with your audio files:
echo    python utils\simplified_inference.py --input_file path\to\audio_file.wav --force_cpu
echo.
echo 2. Check the training results in models\checkpoints\
echo    - training_history.png
echo    - confusion_matrix.png
echo    - roc_curve.png
echo.
echo 3. Review the documentation in GUIDE.md for detailed information
echo. 