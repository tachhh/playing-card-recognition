@echo off
REM Quick Start Menu - Playing Card Recognition Project
REM Auto-activates virtual environment

title Playing Card Recognition - Quick Start

:menu
cls
echo ============================================================
echo   PLAYING CARD RECOGNITION - QUICK START
echo ============================================================
echo.
echo   1. Run Camera (Real-time Detection)
echo   2. Test with Real Images
echo   3. Quick Model Test
echo   4. Train Model
echo   5. Download Dataset
echo.
echo   Q. Quit
echo.
echo ============================================================
echo.

set /p choice="Enter your choice: "

if /i "%choice%"=="1" goto camera
if /i "%choice%"=="2" goto test_real
if /i "%choice%"=="3" goto test_quick
if /i "%choice%"=="4" goto train
if /i "%choice%"=="5" goto download
if /i "%choice%"=="Q" goto quit
if /i "%choice%"=="q" goto quit

echo Invalid choice! Please try again.
timeout /t 2 > nul
goto menu

:camera
cls
echo ========================================
echo  Starting Camera App...
echo ========================================
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
call C:\playing-card-recognition\.venv\Scripts\activate.bat
python camera_simple.py
call C:\playing-card-recognition\.venv\Scripts\deactivate.bat
echo.
pause
goto menu

:test_real
cls
echo ========================================
echo  Testing with Real Images...
echo ========================================
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
call C:\playing-card-recognition\.venv\Scripts\activate.bat
python diagnostics\test_real_image.py
call C:\playing-card-recognition\.venv\Scripts\deactivate.bat
echo.
pause
goto menu

:test_quick
cls
echo ========================================
echo  Quick Model Test...
echo ========================================
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
call C:\playing-card-recognition\.venv\Scripts\activate.bat
python diagnostics\quick_model_test.py
call C:\playing-card-recognition\.venv\Scripts\deactivate.bat
echo.
pause
goto menu

:train
cls
echo ========================================
echo  Training Model...
echo ========================================
echo.
echo WARNING: This will take 2-3 hours!
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto menu
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
call C:\playing-card-recognition\.venv\Scripts\activate.bat
python train_cnn_model.py
call C:\playing-card-recognition\.venv\Scripts\deactivate.bat
echo.
pause
goto menu

:download
cls
echo ========================================
echo  Download Dataset...
echo ========================================
echo.
cd /d C:\playing-card-recognition\playing-card-recognition
call C:\playing-card-recognition\.venv\Scripts\activate.bat
python download_dataset.py
call C:\playing-card-recognition\.venv\Scripts\deactivate.bat
echo.
pause
goto menu

:quit
cls
echo.
echo Thank you for using Playing Card Recognition!
echo.
timeout /t 2 > nul
exit
