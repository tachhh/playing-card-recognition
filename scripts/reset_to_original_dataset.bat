@echo off
echo ========================================
echo Plan B: Reset to Original Dataset
echo ========================================
echo.
echo This will modify train_cnn_model.py to use
echo the original Kaggle dataset instead of merged_dataset
echo.
pause

echo.
echo Resetting train_cnn_model.py...
echo.

call .venv\Scripts\activate

python -c "import re; from pathlib import Path; file = Path('train_cnn_model.py'); content = file.read_text(); new_content = re.sub(r\"dataset_path = project_root / 'data' / 'merged_dataset'\", \"dataset_path = Path.home() / '.cache/kagglehub/datasets/gpiosenka/cards-image-datasetclassification/versions/2'\", content); file.write_text(new_content); print('✅ Reset complete!')"

echo.
echo ========================================
echo Done! train_cnn_model.py now uses original dataset
echo ========================================
echo.
echo Next steps:
echo 1. Run restore_model.bat (if not done yet)
echo 2. Test with run_camera.bat
echo.
pause
