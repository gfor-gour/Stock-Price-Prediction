# Test script for the Stock Price Prediction API

import requests
import json
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 404]  # 404 is OK if no model loaded
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction...")
    
    # Generate sample historical data (simulate stock prices)
    np.random.seed(42)
    base_price = 100.0
    historical_data = []
    for i in range(100):
        price_change = np.random.normal(0, 2)  # Random walk
        base_price += price_change
        historical_data.append(round(base_price, 2))
    
    prediction_data = {
        "symbol": "TEST",
        "historical_data": historical_data,
        "sequence_length": 60
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=prediction_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 503]  # 503 is OK if no model loaded
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Generate sample data for multiple stocks
    stocks_data = []
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
        base_price = 100.0
        historical_data = []
        for i in range(100):
            price_change = np.random.normal(0, 2)
            base_price += price_change
            historical_data.append(round(base_price, 2))
        
        stocks_data.append({
            "symbol": symbol,
            "data": historical_data,
            "sequence_length": 60
        })
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=stocks_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 503]  # 503 is OK if no model loaded
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Stock Price Prediction API Test Suite")
    print("=" * 50)
    
    tests = [
        test_root_endpoint,
        test_health_check,
        test_model_info,
        test_prediction,
        test_batch_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the API server status.")

if __name__ == "__main__":
    main()
