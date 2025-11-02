# run_project.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def load_and_check_data():
    """Load data and check what's available"""
    print("Loading and checking data...")
    
    # Check what data files exist
    data_files = []
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        print(f"Found data files: {data_files}")
    
    # Also check root directory for CSV files
    root_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"Found CSV files in root: {root_files}")
    
    return data_files + root_files

def run_naive_baseline():
    """Implement a simple naive baseline model"""
    print("\n=== Running Naive Baseline Model ===")
    
    # Create sample data if real data isn't available
    dates = pd.date_range('2019-01-01', periods=1000, freq='h')
    np.random.seed(42)
    
    # Synthetic electricity prices (realistic pattern)
    base_price = 50
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))
    daily = 15 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
    noise = np.random.normal(0, 10, len(dates))
    
    prices = base_price + seasonal + daily + noise
    prices = np.maximum(prices, 0)  # No negative prices
    
    # Create DataFrame
    df = pd.DataFrame({'datetime': dates, 'price': prices})
    df.set_index('datetime', inplace=True)
    
    # Naive forecast: next 24 hours = current price
    df['naive_forecast'] = df['price'].shift(24)
    
    # Calculate MAPE
    valid_data = df.dropna()
    mape = np.mean(np.abs((valid_data['price'] - valid_data['naive_forecast']) / valid_data['price'])) * 100
    
    print(f"Naive baseline MAPE: {mape:.2f}%")
    
    # Plot sample of results
    plt.figure(figsize=(12, 6))
    plt.plot(valid_data.index[:168], valid_data['price'][:168], label='Actual', alpha=0.7)
    plt.plot(valid_data.index[:168], valid_data['naive_forecast'][:168], label='Naive Forecast', alpha=0.7)
    plt.title('Naive Baseline Model - Actual vs Predicted')
    plt.legend()
    plt.show()
    
    return mape

def test_machine_learning():
    """Test basic machine learning functionality"""
    print("\n=== Testing Machine Learning Models ===")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Create sample features and target
        np.random.seed(42)
        n_samples = 1000
        
        # Features: hour, day_of_week, month, etc.
        X = np.column_stack([
            np.random.randint(0, 24, n_samples),  # hour
            np.random.randint(1, 8, n_samples),   # day of week
            np.random.randint(1, 13, n_samples),  # month
            np.random.normal(15, 10, n_samples),  # temperature
            np.random.normal(50, 20, n_samples)   # previous price
        ])
        
        # Target: electricity price
        y = 30 + X[:, 0] * 0.5 + X[:, 1] * 2 + X[:, 3] * 0.3 + np.random.normal(0, 5, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Random Forest MAE: {mae:.2f}")
        print("✓ Machine learning functionality working")
        
        # Feature importance
        feature_names = ['hour', 'day_of_week', 'month', 'temperature', 'previous_price']
        importance = model.feature_importances_
        for name, imp in zip(feature_names, importance):
            print(f"  {name}: {imp:.3f}")
            
    except Exception as e:
        print(f"✗ Machine learning test failed: {e}")

def check_deep_learning():
    """Check if deep learning dependencies are available"""
    print("\n=== Checking Deep Learning Setup ===")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow available - Version: {tf.__version__}")
        
        # Test simple model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(24)  # 24-hour forecast
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("✓ Basic neural network model can be created")
        
    except ImportError:
        print("✗ TensorFlow not available")
    
    try:
        import xgboost as xgb
        print("✓ XGBoost available")
    except ImportError:
        print("✗ XGBoost not available")

def main():
    """Main function to run the project"""
    print("=== Electricity Price Prediction Project ===\n")
    
    # Step 1: Check data
    data_files = load_and_check_data()
    
    # Step 2: Run baseline model
    naive_mape = run_naive_baseline()
    
    # Step 3: Test ML capabilities
    test_machine_learning()
    
    # Step 4: Check DL setup
    check_deep_learning()
    
    print("\n=== Project Setup Complete ===")
    print("Next steps:")
    print("1. Check if actual data files are available")
    print("2. Run the Jupyter notebooks in sequence")
    print("3. Start with Initial_eda.ipynb for data exploration")

if __name__ == "__main__":
    main()