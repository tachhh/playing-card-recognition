@echo off
echo ====================================
echo Train with Merged Dataset
echo ====================================
echo.
echo This will train the model with your merged dataset
echo.
echo Checklist:
echo [x] Merged datasets successfully
echo [x] Updated train_cnn_model.py (auto-configured)
echo [ ] Ready to start training
echo.
echo Training will take 1-2 hours on CPU
echo Press Ctrl+C anytime to stop training
echo.
pause
echo.
echo Starting training...
cd /d C:\playing-card-recognition\playing-card-recognition
C:\playing-card-recognition\.venv\Scripts\python.exe train_cnn_model.py
pause
