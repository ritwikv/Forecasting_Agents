"""
Data upload component for Streamlit UI
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class DataUploadComponent:
    """
    Component for handling data upload and validation
    """
    
    def __init__(self):
        """Initialize data upload component"""
        self.required_columns = ['ACD Call Volume Actuals', 'Driver_Forecast']
    
    def render(self) -> Optional[Dict[str, Any]]:
        """
        Render data upload interface
        
        Returns:
            Dictionary with upload results or None
        """
        st.subheader("📁 Upload Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload Excel file with 'ACD Call Volume Actuals' and 'Driver_Forecast' columns"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Preview data
                preview_result = self._preview_data(temp_path)
                
                if preview_result['success']:
                    st.success("✅ File uploaded successfully!")
                    
                    # Display preview
                    st.subheader("Data Preview")
                    st.dataframe(preview_result['preview_data'])
                    
                    # Display validation results
                    validation = preview_result['validation']
                    
                    if validation['is_valid']:
                        st.success("✅ Data validation passed!")
                        
                        # Data summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", validation['total_rows'])
                        with col2:
                            st.metric("Valid Actuals", validation['valid_actuals'])
                        with col3:
                            st.metric("Valid Drivers", validation['valid_drivers'])
                        
                        # Confirm upload button
                        if st.button("✅ Confirm and Use This Data", type="primary"):
                            return {
                                'success': True,
                                'file_path': temp_path,
                                'validation': validation,
                                'preview': preview_result['preview_data']
                            }
                    else:
                        st.error("❌ Data validation failed!")
                        for error in validation['errors']:
                            st.error(f"• {error}")
                        
                        for warning in validation['warnings']:
                            st.warning(f"• {warning}")
                else:
                    st.error(f"❌ Failed to read file: {preview_result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.error(f"File processing error: {str(e)}")
        
        # Sample data option
        st.markdown("---")
        st.subheader("📊 Use Sample Data")
        st.markdown("Don't have data? Use our sample dataset to try the system.")
        
        if st.button("📊 Load Sample Data"):
            return self._load_sample_data()
        
        return None
    
    def _preview_data(self, file_path: str) -> Dict[str, Any]:
        """
        Preview and validate uploaded data
        
        Args:
            file_path: Path to uploaded file
            
        Returns:
            Dictionary with preview results
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Validate data
            validation = self._validate_data(df)
            
            # Create preview (first 10 rows)
            preview_data = df.head(10)
            
            return {
                'success': True,
                'preview_data': preview_data,
                'validation': validation,
                'full_data': df
            }
            
        except Exception as e:
            logger.error(f"Data preview error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate uploaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'total_rows': len(df),
            'valid_actuals': 0,
            'valid_drivers': 0
        }
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            validation['errors'].append(f"Missing required columns: {missing_columns}")
            validation['is_valid'] = False
        
        # Check data quality if columns exist
        if 'ACD Call Volume Actuals' in df.columns:
            actuals_col = df['ACD Call Volume Actuals']
            valid_actuals = actuals_col.notna() & (actuals_col >= 0)
            validation['valid_actuals'] = valid_actuals.sum()
            
            if validation['valid_actuals'] < len(df) * 0.8:
                validation['warnings'].append(
                    f"Only {validation['valid_actuals']}/{len(df)} valid actual values"
                )
            
            if validation['valid_actuals'] < 12:
                validation['errors'].append("Need at least 12 valid actual values for forecasting")
                validation['is_valid'] = False
        
        if 'Driver_Forecast' in df.columns:
            driver_col = df['Driver_Forecast']
            valid_drivers = driver_col.notna() & (driver_col >= 0)
            validation['valid_drivers'] = valid_drivers.sum()
            
            if validation['valid_drivers'] < len(df) * 0.8:
                validation['warnings'].append(
                    f"Only {validation['valid_drivers']}/{len(df)} valid driver values"
                )
        
        # Check for date column
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if not date_columns:
            validation['warnings'].append("No date column detected - will use sequential indexing")
        
        # Check data range
        if len(df) < 12:
            validation['errors'].append("Need at least 12 data points for reliable forecasting")
            validation['is_valid'] = False
        elif len(df) < 24:
            validation['warnings'].append("Limited data - forecast accuracy may be reduced")
        
        return validation
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """
        Load sample data for demonstration
        
        Returns:
            Dictionary with sample data results
        """
        try:
            # Check if sample data file exists
            sample_file = 'Input_Data.xlsx'
            
            if os.path.exists(sample_file):
                # Use existing sample file
                preview_result = self._preview_data(sample_file)
                
                if preview_result['success']:
                    st.success("✅ Sample data loaded successfully!")
                    
                    # Display preview
                    st.subheader("Sample Data Preview")
                    st.dataframe(preview_result['preview_data'])
                    
                    return {
                        'success': True,
                        'file_path': sample_file,
                        'validation': preview_result['validation'],
                        'preview': preview_result['preview_data']
                    }
                else:
                    st.error(f"❌ Failed to load sample data: {preview_result['error']}")
            else:
                # Create sample data on the fly
                sample_data = self._create_sample_data()
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    sample_data.to_excel(tmp_file.name, index=False)
                    temp_path = tmp_file.name
                
                st.success("✅ Sample data created and loaded!")
                
                # Display preview
                st.subheader("Sample Data Preview")
                st.dataframe(sample_data.head(10))
                
                return {
                    'success': True,
                    'file_path': temp_path,
                    'validation': self._validate_data(sample_data),
                    'preview': sample_data.head(10)
                }
                
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            logger.error(f"Sample data error: {str(e)}")
            return None
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for demonstration
        
        Returns:
            Sample DataFrame
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create 3 years of monthly data
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=30*i) for i in range(36)]
        
        # Generate realistic call volume data
        base_volume = 1000
        trend = np.linspace(0, 200, 36)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(36) / 12)
        noise = np.random.normal(0, 50, 36)
        
        actuals = base_volume + trend + seasonal + noise
        actuals = np.maximum(actuals, 0)
        
        # Generate correlated driver forecast
        driver_correlation = 0.7
        driver_noise = np.random.normal(0, 30, 36)
        driver_forecast = (actuals * driver_correlation + 
                          (1 - driver_correlation) * np.mean(actuals) + 
                          driver_noise)
        driver_forecast = np.maximum(driver_forecast, 0)
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'Date': dates,
            'ACD Call Volume Actuals': np.round(actuals).astype(int),
            'Driver_Forecast': np.round(driver_forecast).astype(int)
        })
        
        return sample_data

