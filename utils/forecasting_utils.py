"""
Forecasting utilities for time series analysis and prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ForecastingUtils:
    """
    Utility class for time series forecasting operations
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
    
    def calculate_mape(self, actual: List[float], forecast: List[float]) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MAPE as percentage
        """
        try:
            actual_array = np.array(actual)
            forecast_array = np.array(forecast)
            
            # Remove zero values to avoid division by zero
            mask = actual_array != 0
            if not np.any(mask):
                logger.warning("All actual values are zero, cannot calculate MAPE")
                return float('inf')
            
            mape = mean_absolute_percentage_error(
                actual_array[mask], 
                forecast_array[mask]
            ) * 100
            
            return float(mape)
            
        except Exception as e:
            logger.error(f"Error calculating MAPE: {str(e)}")
            return float('inf')
    
    def calculate_forecast_accuracy_metrics(self, 
                                          actual: List[float], 
                                          forecast: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive forecast accuracy metrics
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            actual_array = np.array(actual)
            forecast_array = np.array(forecast)
            
            # Basic metrics
            mae = np.mean(np.abs(actual_array - forecast_array))
            mse = mean_squared_error(actual_array, forecast_array)
            rmse = np.sqrt(mse)
            mape = self.calculate_mape(actual, forecast)
            
            # Additional metrics
            mean_actual = np.mean(actual_array)
            mean_forecast = np.mean(forecast_array)
            bias = mean_forecast - mean_actual
            bias_percentage = (bias / mean_actual) * 100 if mean_actual != 0 else 0
            
            # Correlation
            correlation = np.corrcoef(actual_array, forecast_array)[0, 1]
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'bias': float(bias),
                'bias_percentage': float(bias_percentage),
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'mean_actual': float(mean_actual),
                'mean_forecast': float(mean_forecast)
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {str(e)}")
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'bias': 0.0,
                'bias_percentage': 0.0,
                'correlation': 0.0,
                'mean_actual': 0.0,
                'mean_forecast': 0.0
            }
    
    def simple_exponential_smoothing(self, 
                                   data: List[float], 
                                   alpha: float = 0.3,
                                   forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Simple exponential smoothing forecast
        
        Args:
            data: Historical data
            alpha: Smoothing parameter (0 < alpha < 1)
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < 2:
                raise ValueError("Need at least 2 data points for exponential smoothing")
            
            # Initialize
            smoothed = np.zeros(n)
            smoothed[0] = data_array[0]
            
            # Apply exponential smoothing
            for i in range(1, n):
                smoothed[i] = alpha * data_array[i] + (1 - alpha) * smoothed[i-1]
            
            # Forecast future periods
            last_smoothed = smoothed[-1]
            forecast = [last_smoothed] * forecast_periods
            
            # Calculate in-sample accuracy
            fitted_values = smoothed[1:]  # Skip first value
            actual_values = data_array[1:]
            accuracy = self.calculate_forecast_accuracy_metrics(
                actual_values.tolist(), 
                fitted_values.tolist()
            )
            
            return {
                'method': 'Simple Exponential Smoothing',
                'parameters': {'alpha': alpha},
                'fitted_values': fitted_values.tolist(),
                'forecast': forecast,
                'accuracy': accuracy,
                'last_smoothed_value': float(last_smoothed)
            }
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing: {str(e)}")
            raise
    
    def linear_trend_forecast(self, 
                            data: List[float],
                            forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Linear trend forecasting
        
        Args:
            data: Historical data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < 3:
                raise ValueError("Need at least 3 data points for linear trend")
            
            # Create time index
            X = np.arange(n).reshape(-1, 1)
            y = data_array
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate fitted values
            fitted_values = model.predict(X)
            
            # Forecast future periods
            future_X = np.arange(n, n + forecast_periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Calculate accuracy
            accuracy = self.calculate_forecast_accuracy_metrics(
                data, fitted_values.tolist()
            )
            
            return {
                'method': 'Linear Trend',
                'parameters': {
                    'slope': float(model.coef_[0]),
                    'intercept': float(model.intercept_)
                },
                'fitted_values': fitted_values.tolist(),
                'forecast': forecast.tolist(),
                'accuracy': accuracy,
                'r_squared': float(model.score(X, y))
            }
            
        except Exception as e:
            logger.error(f"Error in linear trend forecast: {str(e)}")
            raise
    
    def seasonal_naive_forecast(self, 
                              data: List[float],
                              seasonal_period: int = 12,
                              forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Seasonal naive forecasting (repeat seasonal pattern)
        
        Args:
            data: Historical data
            seasonal_period: Length of seasonal cycle
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < seasonal_period:
                raise ValueError(f"Need at least {seasonal_period} data points for seasonal naive")
            
            # Generate fitted values (seasonal naive for in-sample)
            fitted_values = []
            for i in range(n):
                if i < seasonal_period:
                    fitted_values.append(data_array[i])  # Use actual for first season
                else:
                    fitted_values.append(data_array[i - seasonal_period])
            
            # Forecast future periods
            forecast = []
            for i in range(forecast_periods):
                seasonal_index = (n + i) % seasonal_period
                if seasonal_index < len(data_array):
                    forecast.append(data_array[-(seasonal_period - seasonal_index)])
                else:
                    forecast.append(data_array[-1])  # Fallback to last value
            
            # Calculate accuracy (skip first seasonal period for fitted values)
            if n > seasonal_period:
                actual_for_accuracy = data_array[seasonal_period:]
                fitted_for_accuracy = fitted_values[seasonal_period:]
                accuracy = self.calculate_forecast_accuracy_metrics(
                    actual_for_accuracy.tolist(),
                    fitted_for_accuracy
                )
            else:
                accuracy = {'mape': float('inf')}
            
            return {
                'method': 'Seasonal Naive',
                'parameters': {'seasonal_period': seasonal_period},
                'fitted_values': fitted_values,
                'forecast': forecast,
                'accuracy': accuracy
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal naive forecast: {str(e)}")
            raise
    
    def ensemble_forecast(self, 
                         data: List[float],
                         forecast_periods: int = 12,
                         methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ensemble forecasting combining multiple methods
        
        Args:
            data: Historical data
            forecast_periods: Number of periods to forecast
            methods: List of methods to include in ensemble
            
        Returns:
            Dictionary with ensemble forecast results
        """
        try:
            if methods is None:
                methods = ['exponential_smoothing', 'linear_trend', 'seasonal_naive']
            
            forecasts = {}
            weights = {}
            
            # Generate forecasts from different methods
            if 'exponential_smoothing' in methods:
                try:
                    result = self.simple_exponential_smoothing(data, forecast_periods=forecast_periods)
                    forecasts['exponential_smoothing'] = result
                    # Weight based on inverse MAPE (lower MAPE = higher weight)
                    mape = result['accuracy']['mape']
                    weights['exponential_smoothing'] = 1 / (mape + 1) if mape != float('inf') else 0
                except Exception as e:
                    logger.warning(f"Exponential smoothing failed: {str(e)}")
            
            if 'linear_trend' in methods:
                try:
                    result = self.linear_trend_forecast(data, forecast_periods=forecast_periods)
                    forecasts['linear_trend'] = result
                    mape = result['accuracy']['mape']
                    weights['linear_trend'] = 1 / (mape + 1) if mape != float('inf') else 0
                except Exception as e:
                    logger.warning(f"Linear trend failed: {str(e)}")
            
            if 'seasonal_naive' in methods:
                try:
                    result = self.seasonal_naive_forecast(data, forecast_periods=forecast_periods)
                    forecasts['seasonal_naive'] = result
                    mape = result['accuracy']['mape']
                    weights['seasonal_naive'] = 1 / (mape + 1) if mape != float('inf') else 0
                except Exception as e:
                    logger.warning(f"Seasonal naive failed: {str(e)}")
            
            if not forecasts:
                raise ValueError("No forecasting methods succeeded")
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Equal weights if all methods have infinite MAPE
                weights = {k: 1 / len(forecasts) for k in forecasts.keys()}
            
            # Combine forecasts
            ensemble_forecast = np.zeros(forecast_periods)
            ensemble_fitted = np.zeros(len(data))
            
            for method, result in forecasts.items():
                weight = weights[method]
                ensemble_forecast += weight * np.array(result['forecast'])
                ensemble_fitted += weight * np.array(result['fitted_values'])
            
            # Calculate ensemble accuracy
            accuracy = self.calculate_forecast_accuracy_metrics(
                data, ensemble_fitted.tolist()
            )
            
            return {
                'method': 'Ensemble',
                'parameters': {
                    'methods_used': list(forecasts.keys()),
                    'weights': weights
                },
                'fitted_values': ensemble_fitted.tolist(),
                'forecast': ensemble_forecast.tolist(),
                'accuracy': accuracy,
                'individual_forecasts': forecasts
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble forecast: {str(e)}")
            raise
    
    def recommend_forecast_periods(self, 
                                 data_length: int,
                                 target_mape: float = 15.0) -> Dict[str, Any]:
        """
        Recommend appropriate forecast horizon based on data length
        
        Args:
            data_length: Number of historical data points
            target_mape: Target MAPE threshold
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'max_recommended_periods': 0,
            'confidence_level': 'low',
            'reasoning': '',
            'warnings': []
        }
        
        if data_length < 6:
            recommendations['max_recommended_periods'] = 1
            recommendations['confidence_level'] = 'very_low'
            recommendations['reasoning'] = 'Very limited data - only short-term forecast recommended'
            recommendations['warnings'].append('High uncertainty due to insufficient data')
            
        elif data_length < 12:
            recommendations['max_recommended_periods'] = min(3, data_length // 2)
            recommendations['confidence_level'] = 'low'
            recommendations['reasoning'] = 'Limited data - short-term forecasts only'
            recommendations['warnings'].append('Cannot capture seasonal patterns')
            
        elif data_length < 24:
            recommendations['max_recommended_periods'] = min(6, data_length // 3)
            recommendations['confidence_level'] = 'medium'
            recommendations['reasoning'] = 'Moderate data available - medium-term forecasts possible'
            recommendations['warnings'].append('Limited seasonal pattern detection')
            
        elif data_length < 36:
            recommendations['max_recommended_periods'] = min(12, data_length // 3)
            recommendations['confidence_level'] = 'good'
            recommendations['reasoning'] = 'Good data availability - annual forecasts possible'
            
        else:
            recommendations['max_recommended_periods'] = min(24, data_length // 2)
            recommendations['confidence_level'] = 'high'
            recommendations['reasoning'] = 'Excellent data availability - long-term forecasts supported'
        
        return recommendations

