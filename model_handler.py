import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Please install it using: pip install tensorflow")

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handler class for LSTM model operations"""
    
    def __init__(self, model_path: str = "models/lstm_model.h5"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.sequence_length = 60
        
    def load_model(self) -> bool:
        """Load the LSTM model from file"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not available. Cannot load model.")
            return False
            
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_data(self, data: List[float], sequence_length: Optional[int] = None) -> np.ndarray:
        """Preprocess data for LSTM input"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Convert to numpy array
        data_array = np.array(data)
        
        # Take the last sequence_length data points
        if len(data_array) >= sequence_length:
            processed_data = data_array[-sequence_length:]
        else:
            # Pad with zeros if not enough data
            processed_data = np.pad(data_array, (sequence_length - len(data_array), 0), 'constant')
        
        # Reshape for LSTM input (batch_size, timesteps, features)
        processed_data = processed_data.reshape(1, sequence_length, 1)
        
        return processed_data
    
    def predict(self, data: List[float], sequence_length: Optional[int] = None) -> Tuple[float, Optional[float]]:
        """Make prediction using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data, sequence_length)
            
            # Make prediction
            prediction = self.model.predict(processed_data, verbose=0)
            predicted_price = float(prediction[0][0])
            
            # Calculate confidence (you can implement your own logic)
            confidence = self._calculate_confidence(prediction)
            
            return predicted_price, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e
    
    def _calculate_confidence(self, prediction: np.ndarray) -> Optional[float]:
        """Calculate prediction confidence (implement your own logic)"""
        # This is a placeholder - implement your own confidence calculation
        # For example, you could use prediction variance, model uncertainty, etc.
        return None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"model_loaded": False}
        
        try:
            return {
                "model_loaded": True,
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "total_params": self.model.count_params(),
                "model_layers": len(self.model.layers),
                "model_type": type(self.model).__name__
            }
        except Exception as e:
            return {
                "model_loaded": True,
                "error": f"Could not retrieve model info: {str(e)}"
            }
    
    def validate_data(self, data: List[float], sequence_length: Optional[int] = None) -> bool:
        """Validate input data"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if not isinstance(data, list):
            return False
        
        if len(data) < sequence_length:
            return False
        
        # Check if all elements are numeric
        try:
            float_data = [float(x) for x in data]
            return True
        except (ValueError, TypeError):
            return False
