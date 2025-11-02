# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:34:14 2025

@author: Khayreddine04
"""

# corrected_test_dl_setup.py
import numpy as np  # Add this import

print("Testing Deep Learning Setup...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} working")
    
    # Test GPU availability
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        print("✅ GPU available for training")
    else:
        print("ℹ Using CPU for training")
        
    # Test basic model creation
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(24, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(24)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("✅ LSTM model can be created")
    
except Exception as e:
    print(f"❌ TensorFlow issue: {e}")

try:
    import xgboost as xgb
    print("✅ XGBoost working")
    
    # Test XGBoost with proper imports
    data = np.random.randn(100, 5)
    target = data[:, 0] * 2 + np.random.randn(100) * 0.1
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(data, target)
    print("✅ XGBoost model training works")
    
except Exception as e:
    print(f"❌ XGBoost issue: {e}")