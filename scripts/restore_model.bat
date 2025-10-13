@echo off
echo ====================================
echo Restore Model from Backup
echo ====================================
echo.
echo This will restore your previous model
echo.

cd /d C:\playing-card-recognition\playing-card-recognition

REM Check if backup folder exists
if not exist "models\backup" (
    echo Error: No backup folder found!
    echo Please backup your model first using backup_model.bat
    pause
    exit
)

echo Available backups:
echo.
dir /b models\backup\card_classifier_cnn_*.pth
echo.

set /p backup_name="Enter backup filename (or 'latest' for most recent): "

if "%backup_name%"=="latest" (
    REM Get the most recent backup
    for /f "delims=" %%i in ('dir /b /o-d models\backup\card_classifier_cnn_*.pth') do (
        set latest_backup=%%i
        goto :found
    )
)

:found
if "%backup_name%"=="latest" (
    set backup_file=%latest_backup%
) else (
    set backup_file=%backup_name%
)

echo.
echo Restoring: %backup_file%
echo.

REM Extract timestamp from filename
set timestamp=%backup_file:card_classifier_cnn_=%
set timestamp=%timestamp:.pth=%

REM Restore files
if exist "models\backup\card_classifier_cnn_%timestamp%.pth" (
    echo Restoring: card_classifier_cnn.pth
    copy /Y "models\backup\card_classifier_cnn_%timestamp%.pth" "models\card_classifier_cnn.pth"
)

if exist "models\backup\card_classifier_cnn_full_%timestamp%.pth" (
    echo Restoring: card_classifier_cnn_full.pth
    copy /Y "models\backup\card_classifier_cnn_full_%timestamp%.pth" "models\card_classifier_cnn_full.pth"
)

if exist "models\backup\class_mapping_cnn_%timestamp%.json" (
    echo Restoring: class_mapping_cnn.json
    copy /Y "models\backup\class_mapping_cnn_%timestamp%.json" "models\class_mapping_cnn.json"
)

echo.
echo ====================================
echo Restore Complete!
echo ====================================
echo.
echo Old model has been restored
echo You can now use it with camera or test it
echo.
pause
