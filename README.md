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
- 🧪 Built-in testing utilities

## Prerequisites

- **Python 3.9+** (recommended: Python 3.11)
- **pip** (Python package installer)
- **Git** (for cloning the repository)

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd stock-price-prediction-api
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install core dependencies only (without dev tools)
pip install fastapi uvicorn[standard] pydantic pydantic-settings numpy pandas scikit-learn tensorflow python-multipart python-dotenv aiofiles httpx requests
```

### 4. Prepare Your Model

Place your trained LSTM model in one of these locations:
- `models/lstm_model.h5` (preferred)
- `lstm_model.h5` (project root - fallback)

**Model Requirements:**
- File format: `.h5` (Keras/TensorFlow format)
- Input shape: `(batch_size, sequence_length, 1)` where sequence_length is typically 60
- Output shape: `(batch_size, 1)` for single price prediction
- Saved using: `model.save('lstm_model.h5')`

### 5. Run the API

**Option 1: Direct Python execution**
```bash
python main.py
```

**Option 2: Using Uvicorn (recommended for development)**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Using Uvicorn with custom settings**
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload --log-level info
```

### 6. Verify Installation

Open your browser and visit:
- **API Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Detailed Installation Guide

### System Requirements

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.9 | 3.11+ |
| pip | 20.0+ | Latest |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 5GB+ |

### Step-by-Step Installation

#### Windows Installation

1. **Install Python:**
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Create Project Directory:**
   ```cmd
   mkdir stock-prediction-api
   cd stock-prediction-api
   ```

3. **Set up Virtual Environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies:**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### macOS Installation

1. **Install Python (using Homebrew):**
   ```bash
   brew install python
   ```

2. **Create Project Directory:**
   ```bash
   mkdir stock-prediction-api
   cd stock-prediction-api
   ```

3. **Set up Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### Linux Installation

1. **Install Python and pip:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip
   ```

2. **Create Project Directory:**
   ```bash
   mkdir stock-prediction-api
   cd stock-prediction-api
   ```

3. **Set up Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

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
├── setup.bat             # Windows setup script
├── setup.sh              # Unix setup script
├── test_api.py           # API testing script
├── README.md             # This file
├── models/               # Directory for trained models
│   └── lstm_model.h5     # Your trained model (place here)
├── logs/                 # Directory for application logs
└── .env                  # Environment variables (create this)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Application Settings
DEBUG=False
LOG_LEVEL=INFO
APP_NAME=Stock Price Prediction API
APP_VERSION=1.0.0

# Server Settings
HOST=0.0.0.0
PORT=8000

# Model Settings
MODEL_PATH=models/lstm_model.h5
DEFAULT_SEQUENCE_LENGTH=60
MAX_PREDICTION_BATCH_SIZE=100

# CORS Settings
CORS_ORIGINS=["*"]
```

### Model Configuration

The API automatically detects your model in these locations:
1. `models/lstm_model.h5` (primary)
2. `lstm_model.h5` (fallback)

To use a different model path, update the `MODEL_PATH` in your `.env` file.

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Health check and model status |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict stock price for a single stock |
| POST | `/predict-batch` | Predict stock prices for multiple stocks |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-model` | Upload a new model file |
| GET | `/model-info` | Get information about the loaded model |

## Usage Examples

### Single Prediction

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample data (replace with your actual historical prices)
historical_data = [150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0]

# Request payload
payload = {
    "symbol": "AAPL",
    "historical_data": historical_data,
    "sequence_length": 60
}

# Make prediction
response = requests.post(url, json=payload)
result = response.json()

print(f"Predicted price for {result['symbol']}: ${result['predicted_price']:.2f}")
print(f"Timestamp: {result['timestamp']}")
```

### Batch Prediction

```python
import requests

# API endpoint
url = "http://localhost:8000/predict-batch"

# Sample batch data
batch_data = [
    {
        "symbol": "AAPL",
        "data": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
        "sequence_length": 60
    },
    {
        "symbol": "GOOGL",
        "data": [2800.0, 2810.0, 2820.0, 2830.0, 2840.0, 2850.0],
        "sequence_length": 60
    }
]

# Make batch prediction
response = requests.post(url, json=batch_data)
results = response.json()

for prediction in results['predictions']:
    if 'error' not in prediction:
        print(f"{prediction['symbol']}: ${prediction['predicted_price']:.2f}")
    else:
        print(f"{prediction['symbol']}: Error - {prediction['error']}")
```

### Health Check

```python
import requests

# Check API health
response = requests.get("http://localhost:8000/health")
health = response.json()

print(f"Status: {health['status']}")
print(f"Model Loaded: {health['model_loaded']}")
print(f"Message: {health['message']}")
```

## Testing

### Run Built-in Tests

```bash
python test_api.py
```

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "historical_data": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
       "sequence_length": 60
     }'
```

### Load Testing

```bash
# Install Apache Bench (if not installed)
# Ubuntu/Debian: sudo apt install apache2-utils
# macOS: brew install httpd

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 http://localhost:8000/health
```

## Development

### Code Quality Tools

```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

### Adding New Features

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-endpoint
   ```

2. **Make changes and test:**
   ```bash
   python test_api.py
   ```

3. **Commit changes:**
   ```bash
   git add .
   git commit -m "Add new endpoint"
   ```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker Directly

```bash
# Build image
docker build -t stock-prediction-api .

# Run container
docker run -d \
  --name stock-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  stock-prediction-api

# View logs
docker logs stock-api

# Stop container
docker stop stock-api
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `Import "tensorflow" could not be resolved`

**Solution:**
```bash
pip install tensorflow==2.15.0
```

#### 2. Model Not Loading

**Problem:** Model file not found

**Solutions:**
- Ensure your model is named `lstm_model.h5`
- Place it in `models/` directory or project root
- Check file permissions

#### 3. Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Use different port
uvicorn main:app --port 8001

# Or kill existing process
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill
```

#### 4. Memory Issues

**Problem:** Out of memory errors

**Solutions:**
- Reduce batch size in predictions
- Use smaller sequence length
- Increase system RAM
- Use model quantization

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set environment variable
export DEBUG=True

# Or create .env file
echo "DEBUG=True" > .env

# Run with debug
uvicorn main:app --reload --log-level debug
```

### Logs

Check application logs:

```bash
# If using Docker
docker logs stock-api

# If running directly
tail -f logs/app.log
```

## Performance Optimization

### Production Settings

```bash
# Use production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (Linux/macOS)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Model Optimization

- Use TensorFlow Lite for mobile deployment
- Implement model quantization
- Use batch processing for multiple predictions
- Cache frequently used predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

## Changelog

### Version 1.0.0
- Initial release
- FastAPI integration
- LSTM model support
- Docker deployment
- Comprehensive documentation