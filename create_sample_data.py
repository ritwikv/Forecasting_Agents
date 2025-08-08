"""
Create sample Input_Data.xlsx file for testing the forecasting system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create realistic sample data for testing"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create 3 years of monthly data (36 data points)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(36)]
    
    # Generate realistic call volume data with trend and seasonality
    base_volume = 1000
    trend = np.linspace(0, 200, 36)  # Upward trend over time
    seasonal = 100 * np.sin(2 * np.pi * np.arange(36) / 12)  # Annual seasonality
    noise = np.random.normal(0, 50, 36)  # Random noise
    
    actuals = base_volume + trend + seasonal + noise
    actuals = np.maximum(actuals, 0)  # Ensure non-negative values
    
    # Generate driver forecast with correlation to actuals
    driver_correlation = 0.7
    driver_noise = np.random.normal(0, 30, 36)
    driver_forecast = (actuals * driver_correlation + 
                      (1 - driver_correlation) * np.mean(actuals) + 
                      driver_noise)
    driver_forecast = np.maximum(driver_forecast, 0)  # Ensure non-negative values
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'ACD Call Volume Actuals': np.round(actuals).astype(int),
        'Driver_Forecast': np.round(driver_forecast).astype(int)
    })
    
    # Save to Excel
    data.to_excel('Input_Data.xlsx', index=False)
    
    print('Sample Input_Data.xlsx created successfully!')
    print(f'Data shape: {data.shape}')
    print(f'Date range: {data["Date"].min()} to {data["Date"].max()}')
    print(f'Actuals range: {data["ACD Call Volume Actuals"].min()} to {data["ACD Call Volume Actuals"].max()}')
    print(f'Driver forecast range: {data["Driver_Forecast"].min()} to {data["Driver_Forecast"].max()}')
    print(f'Correlation: {data["ACD Call Volume Actuals"].corr(data["Driver_Forecast"]):.3f}')
    
    return data

if __name__ == "__main__":
    create_sample_data()

