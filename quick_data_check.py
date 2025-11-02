# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:39:20 2025

@author: Khayreddine04
"""

# quick_data_check.py
import pandas as pd
import matplotlib.pyplot as plt

print("=== Quick Data Overview ===")

# Load the main dataset
df = pd.read_csv('re_fixed_multivariate_timeseires.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Basic info
print(f"\nData types:")
print(df.dtypes)

# Check for price column and plot
if 'price' in df.columns:
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    
    # Plot first week of data (168 hours)
    plt.figure(figsize=(12, 6))
    plt.plot(df['price'].head(168))
    plt.title('Electricity Prices - First Week (168 hours)')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

# Check data period
if 'datetime' in df.columns or 'date' in df.columns:
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    print(f"\nDate range: {df[date_col].min()} to {df[date_col].max()}")