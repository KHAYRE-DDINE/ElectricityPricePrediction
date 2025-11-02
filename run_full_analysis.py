# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:39:59 2025

@author: Khayreddine04
"""

# run_full_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

print("=== Electricity Price Prediction - Full Analysis ===")

# Load data
print("Loading data...")
multi_df = pd.read_csv('re_fixed_multivariate_timeseires.csv')
price_df = pd.read_csv('refixed_price_only_timeseries.csv')

print(f"Multivariate data: {multi_df.shape}")
print(f"Price-only data: {price_df.shape}")

# Basic analysis
if 'price' in multi_df.columns:
    prices = multi_df['price']
    print(f"\n--- Electricity Price Analysis ---")
    print(f"Total observations: {len(prices)}")
    print(f"Date range: {len(prices) / 24:.1f} days")
    print(f"Average price: {prices.mean():.2f}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    print(f"Volatility (std): {prices.std():.2f}")
    
    # Plot distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(prices.head(168))
    plt.title('First Week of Prices')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.hist(prices, bins=50, alpha=0.7)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Daily pattern (average by hour)
    if len(prices) >= 24:
        daily_pattern = [prices[i::24].mean() for i in range(24)]
        plt.plot(daily_pattern)
        plt.title('Average Daily Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Price')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\n✅ Data loaded successfully!")
print("Next: Run the individual model notebooks for full analysis")