# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:33:25 2025

@author: Khayreddine04
"""

# test_tensorflow.py
print("Testing TensorFlow installation...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} installed successfully!")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU available for training")
    else:
        print("ℹ Using CPU for training")
        
    # Test basic functionality
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("✅ TensorFlow model creation works!")
    
except Exception as e:
    print(f"❌ TensorFlow error: {e}")