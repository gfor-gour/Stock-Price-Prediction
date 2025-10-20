from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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

class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response"""
    predictions: List[dict]

class ModelInfoResponse(BaseModel):
    """Pydantic model for model information response"""
    model_loaded: bool
    model_summary: Optional[str] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    total_params: Optional[int] = None
    timestamp: str
    error: Optional[str] = None
