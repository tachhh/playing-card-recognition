@echo off
echo ====================================
echo Test Trained Model
echo ====================================
echo.
echo This will test your newly trained model
echo on the test dataset (265 images)
echo.
pause
echo.
echo Testing model...
cd /d C:\playing-card-recognition\playing-card-recognition
C:\playing-card-recognition\.venv\Scripts\python.exe test_cnn_model.py
echo.
echo ====================================
echo Test Complete!
echo ====================================
echo.
echo Check the results above to see:
echo - Overall accuracy
echo - Per-class accuracy
echo - Prediction visualization saved
echo.
pause
