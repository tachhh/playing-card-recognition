@echo off
echo ====================================
echo Backup Current Model
echo ====================================
echo.
echo This will backup your current model before training
echo.

cd /d C:\playing-card-recognition\playing-card-recognition

REM Create backup folder
if not exist "models\backup" mkdir "models\backup"

REM Get timestamp
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set timestamp=%mydate%_%mytime%

REM Backup model files
if exist "models\card_classifier_cnn.pth" (
    echo Backing up: card_classifier_cnn.pth
    copy "models\card_classifier_cnn.pth" "models\backup\card_classifier_cnn_%timestamp%.pth"
    echo.
)

if exist "models\card_classifier_cnn_full.pth" (
    echo Backing up: card_classifier_cnn_full.pth
    copy "models\card_classifier_cnn_full.pth" "models\backup\card_classifier_cnn_full_%timestamp%.pth"
    echo.
)

if exist "models\class_mapping_cnn.json" (
    echo Backing up: class_mapping_cnn.json
    copy "models\class_mapping_cnn.json" "models\backup\class_mapping_cnn_%timestamp%.json"
    echo.
)

echo ====================================
echo Backup Complete!
echo ====================================
echo.
echo Backup saved to: models\backup\
echo Timestamp: %timestamp%
echo.
echo You can now safely train a new model
echo If the new model is worse, you can restore the backup
echo.
pause
