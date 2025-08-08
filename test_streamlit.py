"""
Minimal Streamlit app to test if basic functionality works
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Agentic AI Forecasting System - Test",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agentic AI Forecasting System - Test Mode")
    st.write("This is a minimal test to verify Streamlit works correctly.")
    
    # Test basic functionality
    st.header("📊 Basic Test")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    sample_data = pd.DataFrame({
        'Date': dates,
        'ACD Call Volume Actuals': np.random.randint(100, 200, 12),
        'Driver_Forecast': np.random.randint(90, 210, 12)
    })
    
    st.subheader("Sample Data")
    st.dataframe(sample_data)
    
    # Test chart
    st.subheader("Sample Chart")
    st.line_chart(sample_data.set_index('Date'))
    
    # Test imports
    st.header("🧪 Import Test")
    
    try:
        # Test basic imports
        from config import MODEL_CONFIG
        st.success("✅ Config import successful")
        
        from data.data_processor import DataProcessor
        st.success("✅ Data processor import successful")
        
        from utils.forecasting_utils import ForecastingUtils
        st.success("✅ Forecasting utils import successful")
        
        st.success("🎉 All basic imports working!")
        
    except Exception as e:
        st.error(f"❌ Import error: {str(e)}")
        st.info("💡 This helps identify which specific import is causing issues")
    
    st.header("📋 Next Steps")
    st.info("""
    If this test page loads successfully, the basic Streamlit setup is working.
    
    To run the full application:
    1. Make sure all dependencies are installed: `pip install -r requirements.txt`
    2. Download the Phi-3 model file
    3. Run: `python simple_launch.py`
    """)

if __name__ == "__main__":
    main()

