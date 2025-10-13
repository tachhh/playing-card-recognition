@echo off
REM Run Camera Simple - Playing Card Recognition
REM This script automatically activates virtual environment and runs camera

echo ========================================
echo  Playing Card Recognition - Camera App
echo ========================================
echo.

REM Change to project directory
cd /d C:\playing-card-recognition\playing-card-recognition

REM Activate virtual environment
call C:\playing-card-recognition\.venv\Scripts\activate.bat

REM Check if activation succeeded
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Run camera script
echo Starting camera...
echo.
echo Controls:
echo   Press 'Q' to quit
echo   Press 'S' to save frame
echo   Press 'F' to toggle detection mode
echo.
echo ========================================
echo.

python camera_simple.py

REM Deactivate virtual environment
deactivate

echo.
echo ========================================
echo Program ended
echo ========================================
pause
