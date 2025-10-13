@echo off
echo ====================================
echo Check New Dataset Format
echo ====================================
cd /d C:\playing-card-recognition\playing-card-recognition
C:\playing-card-recognition\.venv\Scripts\python.exe convert_dataset.py
pause
