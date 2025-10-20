from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import logging

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Please install it using: pip install tensorflow")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="API for predicting stock prices using LSTM deep learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None

class StockData(BaseModel):
    """Pydantic model for stock data input"""
    symbol: str
    data: List[float]  # Historical price data
    sequence_length: Optional[int] = 60  # Default sequence length for LSTM

class PredictionRequest(BaseModel):
    """Pydantic model for prediction request"""
    symbol: str
    historical_data: List[float]
    sequence_length: Optional[int] = 60

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    symbol: str
    predicted_price: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: Optional[str] = None

class HealthResponse(BaseModel):
    """Pydantic model for health check response"""
    status: str
    message: str
    timestamp: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global model
    try:
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Model loading skipped.")
            return
            
        # Try to load the model from the models directory; fallback to project root
        model_path = "models/lstm_model.h5"
        root_model_path = "lstm_model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        elif os.path.exists(root_model_path):
            model = tf.keras.models.load_model(root_model_path)
            logger.info(f"Model loaded successfully from {root_model_path}")
        else:
            logger.warning(
                f"Model file not found at {model_path} or {root_model_path}. Please place your trained model."
            )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Predict stock price using LSTM model"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and is properly formatted."
        )
    
    try:
        # Validate input data
        if len(request.historical_data) < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data. Need at least {request.sequence_length} data points."
            )
        
        # Prepare data for prediction
        data = np.array(request.historical_data[-request.sequence_length:])
        data = data.reshape(1, request.sequence_length, 1)  # Reshape for LSTM input
        
        # Make prediction
        prediction = model.predict(data, verbose=0)
        predicted_price = float(prediction[0][0])
        
        # Calculate confidence (you can implement your own confidence calculation)
        confidence = None  # Implement confidence calculation based on your model
        
        return PredictionResponse(
            symbol=request.symbol,
            predicted_price=predicted_price,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version="1.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(stock_data: List[StockData]):
    """Predict stock prices for multiple stocks"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and is properly formatted."
        )
    
    results = []
    
    for stock in stock_data:
        try:
            if len(stock.data) < stock.sequence_length:
                results.append({
                    "symbol": stock.symbol,
                    "error": f"Not enough historical data. Need at least {stock.sequence_length} data points."
                })
                continue
            
            # Prepare data for prediction
            data = np.array(stock.data[-stock.sequence_length:])
            data = data.reshape(1, stock.sequence_length, 1)
            
            # Make prediction
            prediction = model.predict(data, verbose=0)
            predicted_price = float(prediction[0][0])
            
            results.append({
                "symbol": stock.symbol,
                "predicted_price": predicted_price,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            results.append({
                "symbol": stock.symbol,
                "error": str(e)
            })
    
    return {"predictions": results}

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new model file"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save uploaded file
        file_path = f"models/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load the new model
        global model
        model = tf.keras.models.load_model(file_path)
        
        return {
            "message": "Model uploaded and loaded successfully",
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model upload failed: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=404,
            detail="No model loaded"
        )
    
    try:
        return {
            "model_loaded": True,
            "model_summary": model.summary(),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "model_loaded": True,
            "error": f"Could not retrieve model info: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
