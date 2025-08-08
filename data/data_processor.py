"""
Data processor for handling Excel input and preparing time series data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings

from config import DATA_CONFIG, LOGGING_CONFIG
from .validators import DataValidator

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles Excel data ingestion, validation, and preprocessing for forecasting
    """
    
    def __init__(self, file_path: str = None):
        """
        Initialize DataProcessor
        
        Args:
            file_path: Path to Excel file. If None, uses default from config
        """
        self.file_path = file_path or DATA_CONFIG["input_file"]
        self.validator = DataValidator()
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Excel file not found: {self.file_path}")
            
            # Load Excel file
            logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_excel(
                self.file_path, 
                sheet_name=DATA_CONFIG["sheet_name"]
            )
            
            # Validate required columns
            required_cols = DATA_CONFIG["required_columns"]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Successfully loaded {len(self.data)} rows of data")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data for forecasting
        
        Returns:
            Preprocessed DataFrame ready for forecasting
        """
        if self.data is None:
            self.load_data()
        
        logger.info("Starting data preprocessing")
        
        # Create a copy for processing
        processed = self.data.copy()
        
        # Handle date column
        processed = self._process_dates(processed)
        
        # Clean and validate numeric columns
        processed = self._clean_numeric_columns(processed)
        
        # Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Sort by date
        if DATA_CONFIG["date_column"] in processed.columns:
            processed = processed.sort_values(DATA_CONFIG["date_column"])
        
        # Validate processed data
        self.validator.validate_processed_data(processed)
        
        self.processed_data = processed
        logger.info("Data preprocessing completed successfully")
        
        return processed
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process date column and create time-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed dates
        """
        date_col = DATA_CONFIG["date_column"]
        
        if date_col in df.columns:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Create additional time features
            df['Year'] = df[date_col].dt.year
            df['Month'] = df[date_col].dt.month
            df['Quarter'] = df[date_col].dt.quarter
            df['YearMonth'] = df[date_col].dt.to_period('M')
            
        else:
            # If no date column, try to infer from index or create one
            logger.warning(f"Date column '{date_col}' not found. Attempting to create date index.")
            
            # Try to create a monthly date range
            if len(df) > 0:
                start_date = datetime.now() - timedelta(days=30 * len(df))
                date_range = pd.date_range(
                    start=start_date, 
                    periods=len(df), 
                    freq='M'
                )
                df[date_col] = date_range
                df['Year'] = df[date_col].dt.year
                df['Month'] = df[date_col].dt.month
                df['Quarter'] = df[date_col].dt.quarter
                df['YearMonth'] = df[date_col].dt.to_period('M')
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate numeric columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned numeric columns
        """
        numeric_cols = DATA_CONFIG["required_columns"]
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove negative values (assuming call volumes can't be negative)
                df.loc[df[col] < 0, col] = np.nan
                
                # Log any extreme outliers
                if df[col].notna().sum() > 0:
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    outliers = df[(df[col] > q99) | (df[col] < q01)][col].count()
                    if outliers > 0:
                        logger.warning(f"Found {outliers} potential outliers in {col}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        numeric_cols = DATA_CONFIG["required_columns"]
        
        for col in numeric_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    logger.info(f"Handling {missing_count} missing values in {col}")
                    
                    # For time series, use forward fill then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still missing, use median
                    if df[col].isna().sum() > 0:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        logger.warning(f"Used median ({median_val}) for remaining missing values in {col}")
        
        return df
    
    def get_forecast_data(self) -> Dict:
        """
        Get data formatted for forecasting agents
        
        Returns:
            Dictionary with formatted data for agents
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        # Extract required columns
        actuals_col = "ACD Call Volume Actuals"
        driver_col = "Driver_Forecast"
        date_col = DATA_CONFIG["date_column"]
        
        forecast_data = {
            'dates': self.processed_data[date_col].tolist() if date_col in self.processed_data.columns else None,
            'actuals': self.processed_data[actuals_col].tolist(),
            'driver_forecast': self.processed_data[driver_col].tolist(),
            'data_points': len(self.processed_data),
            'date_range': {
                'start': self.processed_data[date_col].min() if date_col in self.processed_data.columns else None,
                'end': self.processed_data[date_col].max() if date_col in self.processed_data.columns else None
            },
            'statistics': {
                'actuals_mean': self.processed_data[actuals_col].mean(),
                'actuals_std': self.processed_data[actuals_col].std(),
                'driver_mean': self.processed_data[driver_col].mean(),
                'driver_std': self.processed_data[driver_col].std(),
                'correlation': self.processed_data[actuals_col].corr(self.processed_data[driver_col])
            }
        }
        
        return forecast_data
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the data
        
        Returns:
            Dictionary with data summary
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        summary = {
            'total_records': len(self.processed_data),
            'date_range': {
                'start': self.processed_data[DATA_CONFIG["date_column"]].min() if DATA_CONFIG["date_column"] in self.processed_data.columns else None,
                'end': self.processed_data[DATA_CONFIG["date_column"]].max() if DATA_CONFIG["date_column"] in self.processed_data.columns else None
            },
            'columns': list(self.processed_data.columns),
            'missing_values': self.processed_data.isnull().sum().to_dict(),
            'data_types': self.processed_data.dtypes.to_dict()
        }
        
        # Add statistics for numeric columns
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        summary['statistics'] = self.processed_data[numeric_cols].describe().to_dict()
        
        return summary

