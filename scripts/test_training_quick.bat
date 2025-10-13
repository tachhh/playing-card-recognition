@echo off
echo ========================================
echo Quick Training Test (5 epochs)
echo ========================================
echo.
echo This will test training with ORIGINAL dataset
echo to verify if code is working properly
echo.
pause

call .venv\Scripts\activate
python test_training_quick.py
pause
