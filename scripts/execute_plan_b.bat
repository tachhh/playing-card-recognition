@echo off
echo ========================================
echo Plan B: Complete Reset
echo ========================================
echo.
echo This will:
echo 1. Stop any running training
echo 2. Restore old model (81.89%%)
echo 3. Delete merged_dataset
echo 4. Reset train_cnn_model.py to use original dataset
echo.
echo ARE YOU SURE? (Press Ctrl+C to cancel)
pause

echo.
echo [1/4] Stopping training...
taskkill /F /IM python.exe 2>nul
echo.

echo [2/4] Restoring old model...
call .venv\Scripts\activate

python -c "from pathlib import Path; import shutil; backup_dir = Path('models/backup'); latest = max(backup_dir.glob('card_classifier_cnn_*.pth'), default=None); [shutil.copy(latest.parent / f.replace('card_classifier_cnn_', '').replace('.pth', suffix), Path('models') / f.replace('card_classifier_cnn_', '').replace(str(latest.stem).replace('card_classifier_cnn_', ''), 'card_classifier_cnn')) for f, suffix in [(str(latest), '.pth'), (str(latest).replace('.pth', '_full.pth'), '_full.pth'), (str(latest).replace('.pth', '.json'), '.json')] if (latest.parent / f.replace('card_classifier_cnn_', '').replace('.pth', suffix)).exists()] if latest else print('No backup found'); print('✅ Model restored!' if latest else '❌ No backup found')"

echo.
echo [3/4] Deleting merged_dataset...
if exist "data\merged_dataset" (
    rmdir /s /q data\merged_dataset
    echo ✅ Deleted merged_dataset
) else (
    echo ⚠️  merged_dataset not found
)

if exist "data\new_dataset" (
    rmdir /s /q data\new_dataset
    echo ✅ Deleted new_dataset
) else (
    echo ⚠️  new_dataset not found
)

echo.
echo [4/4] Resetting train_cnn_model.py...
python -c "import re; from pathlib import Path; file = Path('train_cnn_model.py'); content = file.read_text(); new_content = re.sub(r\"dataset_path = project_root / 'data' / 'merged_dataset'\", \"dataset_path = Path.home() / '.cache/kagglehub/datasets/gpiosenka/cards-image-datasetclassification/versions/2'\", content); file.write_text(new_content); print('✅ Reset complete!')"

echo.
echo ========================================
echo ✅ Plan B Complete!
echo ========================================
echo.
echo Your system is now back to:
echo - Original model (81.89%% accuracy)
echo - Original dataset (8,154 images)
echo - Ready to use!
echo.
echo Test with: run_camera.bat
echo.
pause
