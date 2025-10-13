@echo off
echo ====================================
echo Compare Models
echo ====================================
echo.
echo This will compare accuracy between
echo your old and new models
echo.
pause
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
C:\playing-card-recognition\.venv\Scripts\python.exe compare_models.py
pause
