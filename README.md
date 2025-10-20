# Stock Price Prediction API

A FastAPI-based web service for predicting stock prices using LSTM deep learning models.

## Features

- 🚀 FastAPI framework for high-performance API
- 🧠 LSTM model integration for stock price prediction
- 📊 Batch prediction support
- 🔄 Model upload and management
- 📈 Health check endpoints
- 🐳 Docker containerization support
- 📝 Comprehensive API documentation

## Project Structure

```
stock-prediction-api/
├── main.py                 # Main FastAPI application
├── models.py              # Pydantic data models
├── model_handler.py       # Model loading and prediction logic
├── config.py              # Application configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .env                  # Environment variables (create this)
├── models/               # Directory for trained models
└── logs/                 # Directory for application logs
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Environment File

Create a `.env` file in the project root:

```env
DEBUG=False
LOG_LEVEL=INFO
MODEL_PATH=models/lstm_model.h5
DEFAULT_SEQUENCE_LENGTH=60
```

### 3. Prepare Your Model

Place your trained LSTM model in the `models/` directory as `lstm_model.h5`. The model should be saved using:

```python
# In your Google Colab training script
model.save('lstm_model.h5')
```

### 4. Run the Application

#### Option 1: Direct Python
```bash
python main.py
```

#### Option 2: Using Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 3: Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t stock-prediction-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models stock-prediction-api
```

## API Endpoints

### Health Check
- **GET** `/health` - Check API health and model status

### Predictions
- **POST** `/predict` - Predict stock price for a single stock
- **POST** `/predict-batch` - Predict stock prices for multiple stocks

### Model Management
- **POST** `/upload-model` - Upload a new model file
- **GET** `/model-info` - Get information about the loaded model

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation (ReDoc)

## Usage Examples

### Single Prediction

```python
import requests

# Prepare data
data = {
    "symbol": "AAPL",
    "historical_data": [150.0, 151.0, 152.0, ...],  # Your historical price data
    "sequence_length": 60
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Predicted price for {result['symbol']}: ${result['predicted_price']:.2f}")
```

### Batch Prediction

```python
import requests

# Prepare batch data
batch_data = [
    {
        "symbol": "AAPL",
        "data": [150.0, 151.0, 152.0, ...],
        "sequence_length": 60
    },
    {
        "symbol": "GOOGL",
        "data": [2800.0, 2810.0, 2820.0, ...],
        "sequence_length": 60
    }
]

# Make batch prediction
response = requests.post("http://localhost:8000/predict-batch", json=batch_data)
results = response.json()

for prediction in results['predictions']:
    print(f"{prediction['symbol']}: ${prediction['predicted_price']:.2f}")
```

### Upload New Model

```python
import requests

# Upload new model file
with open('new_model.h5', 'rb') as f:
    files = {'file': f}
    response = requests.post("http://localhost:8000/upload-model", files=files)
    
print(response.json())
```

## Model Requirements

Your LSTM model should:

1. **Input Shape**: `(batch_size, sequence_length, 1)` where sequence_length is typically 60
2. **Output Shape**: `(batch_size, 1)` - single price prediction
3. **File Format**: Saved as `.h5` file using `model.save()`
4. **Data Preprocessing**: The API expects raw price data and handles normalization internally

### Example Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

## Configuration

The application can be configured through environment variables:

- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MODEL_PATH`: Path to the model file (default: models/lstm_model.h5)
- `DEFAULT_SEQUENCE_LENGTH`: Default sequence length for predictions (default: 60)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## Production Deployment

### Using Docker

1. Build the Docker image:
```bash
docker build -t stock-prediction-api .
```

2. Run with proper volume mounts:
```bash
docker run -d \
  --name stock-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  stock-prediction-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Environment Variables for Production

Create a `.env` file with production settings:

```env
DEBUG=False
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://yourdomain.com"]
```

## Monitoring and Logging

The application includes:

- Health check endpoint for monitoring
- Structured logging
- Error handling and validation
- Model status tracking

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure the model file exists in the `models/` directory
2. **Prediction errors**: Check that input data has enough historical points
3. **Memory issues**: Consider reducing batch size or sequence length
4. **CORS errors**: Configure CORS origins properly for production

### Logs

Check application logs for detailed error information:

```bash
# If using Docker
docker logs stock-api

# If running directly
tail -f logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
