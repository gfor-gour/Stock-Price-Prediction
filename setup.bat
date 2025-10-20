@echo off
echo Installing Stock Price Prediction API Dependencies...
echo.

echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Place your trained LSTM model (.h5 file) in the 'models' folder
echo 2. Rename it to 'lstm_model.h5' or update the model path in config.py
echo 3. Run the API: python main.py
echo.
echo Your model should be placed at: F:\Stock Price Prediction\models\lstm_model.h5
echo.
pause
