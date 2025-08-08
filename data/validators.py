"""
Data validation utilities for the forecasting system
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime

from config import DATA_CONFIG, AGENT_CONFIG

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data quality and integrity for forecasting
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate raw data from Excel file
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results['errors'].append("DataFrame is empty")
            results['is_valid'] = False
            return results
        
        # Check required columns
        required_cols = DATA_CONFIG["required_columns"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['is_valid'] = False
        
        # Check data types and basic statistics
        for col in required_cols:
            if col in df.columns:
                # Check if column has any non-null values
                non_null_count = df[col].notna().sum()
                if non_null_count == 0:
                    results['errors'].append(f"Column '{col}' has no valid data")
                    results['is_valid'] = False
                elif non_null_count < len(df) * 0.5:
                    results['warnings'].append(f"Column '{col}' has {non_null_count}/{len(df)} valid values")
        
        # Check date column if specified
        date_col = DATA_CONFIG.get("date_column")
        if date_col and date_col in df.columns:
            try:
                pd.to_datetime(df[date_col], errors='coerce')
                results['info'].append(f"Date column '{date_col}' found and parseable")
            except Exception as e:
                results['warnings'].append(f"Date column '{date_col}' may have parsing issues: {str(e)}")
        
        self.validation_results['raw_data'] = results
        return results
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed data ready for forecasting
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check minimum data points for forecasting
        min_points = AGENT_CONFIG["forecasting_agent"]["min_data_points"]
        if len(df) < min_points:
            results['errors'].append(f"Insufficient data points: {len(df)} < {min_points}")
            results['is_valid'] = False
        
        # Validate numeric columns
        numeric_cols = DATA_CONFIG["required_columns"]
        for col in numeric_cols:
            if col in df.columns:
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    results['errors'].append(f"Column '{col}' contains {inf_count} infinite values")
                    results['is_valid'] = False
                
                # Check for negative values (assuming call volumes should be non-negative)
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    results['warnings'].append(f"Column '{col}' contains {neg_count} negative values")
                
                # Check variance (too low variance might indicate data quality issues)
                if df[col].var() < 1e-10:
                    results['warnings'].append(f"Column '{col}' has very low variance")
        
        # Check for data continuity (gaps in time series)
        date_col = DATA_CONFIG.get("date_column")
        if date_col and date_col in df.columns:
            date_gaps = self._check_date_continuity(df[date_col])
            if date_gaps:
                results['warnings'].append(f"Found {len(date_gaps)} gaps in date sequence")
        
        # Check correlation between actuals and driver forecast
        actuals_col = "ACD Call Volume Actuals"
        driver_col = "Driver_Forecast"
        
        if actuals_col in df.columns and driver_col in df.columns:
            correlation = df[actuals_col].corr(df[driver_col])
            if abs(correlation) < 0.1:
                results['warnings'].append(f"Low correlation ({correlation:.3f}) between actuals and driver forecast")
            else:
                results['info'].append(f"Correlation between actuals and driver: {correlation:.3f}")
        
        self.validation_results['processed_data'] = results
        return results
    
    def validate_forecast_input(self, forecast_data: Dict) -> Dict:
        """
        Validate data prepared for forecasting agents
        
        Args:
            forecast_data: Dictionary with forecast input data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check required keys
        required_keys = ['actuals', 'driver_forecast', 'data_points']
        missing_keys = [key for key in required_keys if key not in forecast_data]
        
        if missing_keys:
            results['errors'].append(f"Missing required keys: {missing_keys}")
            results['is_valid'] = False
            return results
        
        # Validate data points count
        min_points = AGENT_CONFIG["forecasting_agent"]["min_data_points"]
        if forecast_data['data_points'] < min_points:
            results['errors'].append(f"Insufficient data points: {forecast_data['data_points']} < {min_points}")
            results['is_valid'] = False
        
        # Validate actuals data
        actuals = forecast_data['actuals']
        if not actuals or len(actuals) == 0:
            results['errors'].append("No actual values provided")
            results['is_valid'] = False
        else:
            # Check for valid numeric values
            valid_actuals = [x for x in actuals if x is not None and not np.isnan(x)]
            if len(valid_actuals) < len(actuals) * 0.8:
                results['warnings'].append(f"Only {len(valid_actuals)}/{len(actuals)} actual values are valid")
        
        # Validate driver forecast data
        driver_forecast = forecast_data['driver_forecast']
        if not driver_forecast or len(driver_forecast) == 0:
            results['warnings'].append("No driver forecast values provided")
        else:
            valid_drivers = [x for x in driver_forecast if x is not None and not np.isnan(x)]
            if len(valid_drivers) < len(driver_forecast) * 0.8:
                results['warnings'].append(f"Only {len(valid_drivers)}/{len(driver_forecast)} driver forecast values are valid")
        
        # Check data alignment
        if len(actuals) != len(driver_forecast):
            results['warnings'].append(f"Actuals ({len(actuals)}) and driver forecast ({len(driver_forecast)}) have different lengths")
        
        self.validation_results['forecast_input'] = results
        return results
    
    def _check_date_continuity(self, dates: pd.Series) -> List[Tuple]:
        """
        Check for gaps in date sequence
        
        Args:
            dates: Series of dates
            
        Returns:
            List of tuples representing gaps (start_date, end_date)
        """
        gaps = []
        
        if len(dates) < 2:
            return gaps
        
        # Sort dates
        sorted_dates = dates.sort_values()
        
        # Check for monthly gaps (assuming monthly data)
        for i in range(1, len(sorted_dates)):
            current_date = sorted_dates.iloc[i]
            prev_date = sorted_dates.iloc[i-1]
            
            # Calculate expected next date (assuming monthly frequency)
            if prev_date.month == 12:
                expected_date = prev_date.replace(year=prev_date.year + 1, month=1)
            else:
                expected_date = prev_date.replace(month=prev_date.month + 1)
            
            # Check if there's a gap
            if current_date != expected_date:
                gaps.append((prev_date, current_date))
        
        return gaps
    
    def get_validation_summary(self) -> Dict:
        """
        Get summary of all validation results
        
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'overall_valid': True,
            'total_errors': 0,
            'total_warnings': 0,
            'validations_performed': list(self.validation_results.keys())
        }
        
        for validation_type, results in self.validation_results.items():
            if not results['is_valid']:
                summary['overall_valid'] = False
            
            summary['total_errors'] += len(results['errors'])
            summary['total_warnings'] += len(results['warnings'])
        
        summary['details'] = self.validation_results
        
        return summary
    
    def recommend_forecast_horizon(self, data_points: int) -> Dict:
        """
        Recommend appropriate forecast horizon based on available data
        
        Args:
            data_points: Number of available data points
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'max_recommended_months': 0,
            'confidence_level': 'low',
            'reasoning': '',
            'warnings': []
        }
        
        # Rule-based recommendations
        if data_points < 12:
            recommendations['max_recommended_months'] = max(1, data_points // 4)
            recommendations['confidence_level'] = 'very_low'
            recommendations['reasoning'] = 'Very limited historical data available'
            recommendations['warnings'].append('Forecasts will have high uncertainty')
            
        elif data_points < 24:
            recommendations['max_recommended_months'] = min(6, data_points // 2)
            recommendations['confidence_level'] = 'low'
            recommendations['reasoning'] = 'Limited historical data for seasonal patterns'
            recommendations['warnings'].append('May not capture seasonal variations well')
            
        elif data_points < 36:
            recommendations['max_recommended_months'] = min(12, data_points // 3)
            recommendations['confidence_level'] = 'medium'
            recommendations['reasoning'] = 'Sufficient data for basic forecasting'
            
        else:
            recommendations['max_recommended_months'] = min(24, data_points // 2)
            recommendations['confidence_level'] = 'high'
            recommendations['reasoning'] = 'Good historical data available for reliable forecasting'
        
        return recommendations

