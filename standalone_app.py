"""
Standalone Streamlit app with minimal imports to test functionality
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_sample_data():
    """Create sample data for testing"""
    dates = pd.date_range('2023-01-01', periods=24, freq='M')
    np.random.seed(42)  # For reproducible results
    
    # Create realistic call volume data with trend and seasonality
    base_volume = 1000
    trend = np.linspace(0, 200, 24)
    seasonal = 100 * np.sin(2 * np.pi * np.arange(24) / 12)
    noise = np.random.normal(0, 50, 24)
    
    actuals = base_volume + trend + seasonal + noise
    
    # Create correlated driver forecast
    driver_correlation = 0.8
    driver_noise = np.random.normal(0, 30, 24)
    drivers = actuals * driver_correlation + driver_noise
    
    return pd.DataFrame({
        'Date': dates,
        'ACD Call Volume Actuals': actuals.astype(int),
        'Driver_Forecast': drivers.astype(int)
    })

def simple_forecast(data, periods=6):
    """Simple forecasting method for testing"""
    actuals = data['ACD Call Volume Actuals'].values
    
    # Simple linear trend + seasonal naive
    if len(actuals) >= 12:
        # Calculate trend
        x = np.arange(len(actuals))
        trend_coef = np.polyfit(x, actuals, 1)[0]
        
        # Calculate seasonal component (last 12 months)
        seasonal = actuals[-12:]
        
        # Generate forecast
        forecast = []
        for i in range(periods):
            trend_component = actuals[-1] + trend_coef * (i + 1)
            seasonal_component = seasonal[i % 12] - np.mean(seasonal)
            forecast.append(trend_component + seasonal_component * 0.3)
    else:
        # Simple trend if not enough data for seasonality
        trend = np.mean(np.diff(actuals[-6:]))
        forecast = [actuals[-1] + trend * (i + 1) for i in range(periods)]
    
    return np.array(forecast)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Agentic AI Forecasting System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agentic AI Forecasting System")
    st.markdown("**Standalone Version - Testing Core Functionality**")
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Data Upload", "🔮 Forecasting", "🔧 Tweaking", "💡 Explanation"]
    )
    
    if page == "📊 Data Upload":
        st.header("📊 Data Upload & Validation")
        
        # Option to use sample data or upload
        data_option = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload Excel File"]
        )
        
        if data_option == "Use Sample Data":
            st.info("Using generated sample data for testing")
            data = create_sample_data()
            st.session_state['data'] = data
            
        else:
            uploaded_file = st.file_uploader(
                "Upload Excel file", 
                type=['xlsx', 'xls'],
                help="File should contain 'ACD Call Volume Actuals' and 'Driver_Forecast' columns"
            )
            
            if uploaded_file:
                try:
                    data = pd.read_excel(uploaded_file)
                    
                    # Validate columns
                    required_cols = ['ACD Call Volume Actuals', 'Driver_Forecast']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                        st.info("Required columns: ACD Call Volume Actuals, Driver_Forecast")
                    else:
                        st.success("✅ Data uploaded successfully!")
                        st.session_state['data'] = data
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Display data if available
        if 'data' in st.session_state:
            data = st.session_state['data']
            
            st.subheader("📈 Data Preview")
            st.dataframe(data.head(10))
            
            st.subheader("📊 Data Visualization")
            fig = go.Figure()
            
            if 'Date' in data.columns:
                x_axis = data['Date'].tolist()
            else:
                x_axis = list(range(len(data)))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=data['ACD Call Volume Actuals'].tolist(),
                mode='lines+markers',
                name='Actuals',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=data['Driver_Forecast'].tolist(),
                mode='lines+markers',
                name='Driver Forecast',
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title="Historical Data",
                xaxis_title="Time Period",
                yaxis_title="Call Volume",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points", len(data))
                st.metric("Avg Actuals", f"{data['ACD Call Volume Actuals'].mean():.0f}")
            
            with col2:
                st.metric("Latest Actual", f"{data['ACD Call Volume Actuals'].iloc[-1]:.0f}")
                st.metric("Avg Driver", f"{data['Driver_Forecast'].mean():.0f}")
    
    elif page == "🔮 Forecasting":
        st.header("🔮 Agent 1: Forecasting")
        
        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first in the Data Upload section")
            return
        
        data = st.session_state['data']
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_periods = st.slider("Forecast Periods", 1, 12, 6)
        with col2:
            method = st.selectbox("Method", ["Simple Trend", "Seasonal Naive", "Ensemble"])
        
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Simple forecast for testing
                forecast = simple_forecast(data, forecast_periods)
                
                # Store in session state
                st.session_state['agent1_forecast'] = forecast
                st.session_state['forecast_periods'] = forecast_periods
                
                st.success("✅ Forecast generated successfully!")
        
        # Display forecast if available
        if 'agent1_forecast' in st.session_state:
            forecast = st.session_state['agent1_forecast']
            
            st.subheader("📊 Forecast Results")
            
            # Create forecast dataframe
            last_date = data['Date'].iloc[-1] if 'Date' in data.columns else len(data)
            if 'Date' in data.columns:
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=len(forecast),
                    freq='M'
                )
            else:
                forecast_dates = range(len(data), len(data) + len(forecast))
            
            forecast_df = pd.DataFrame({
                'Period': forecast_dates,
                'Agent1_Forecast': forecast.astype(int)
            })
            
            st.dataframe(forecast_df)
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            if 'Date' in data.columns:
                hist_x = data['Date'].tolist()
                forecast_x = forecast_dates.tolist()
            else:
                hist_x = list(range(len(data)))
                forecast_x = list(range(len(data), len(data) + len(forecast)))
            
            fig.add_trace(go.Scatter(
                x=hist_x,
                y=data['ACD Call Volume Actuals'].tolist(),
                mode='lines+markers',
                name='Historical Actuals',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast.tolist(),
                mode='lines+markers',
                name='Agent 1 Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Forecast Results",
                xaxis_title="Time Period",
                yaxis_title="Call Volume",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Forecast", f"{forecast.mean():.0f}")
            with col2:
                st.metric("Min Forecast", f"{forecast.min():.0f}")
            with col3:
                st.metric("Max Forecast", f"{forecast.max():.0f}")
    
    elif page == "🔧 Tweaking":
        st.header("🔧 Agent 2: Forecast Tweaking")
        
        if 'agent1_forecast' not in st.session_state:
            st.warning("⚠️ Please generate a forecast first in the Forecasting section")
            return
        
        st.info("💡 Provide instructions to adjust the forecast (up to 300 words)")
        
        # Example instructions
        with st.expander("📝 Example Instructions"):
            st.write("""
            **Example 1:** "The forecast for December is higher than the forecasted drivers for that month. Please reduce it by 15%."
            
            **Example 2:** "The forecast for July is much lower than the average by 35%. Increase it to align with business expectations."
            
            **Example 3:** "The forecast shows too much volatility. Please smooth it out while maintaining the overall trend."
            """)
        
        # Instruction input
        instruction = st.text_area(
            "Tweaking Instructions:",
            placeholder="Enter your instructions here...",
            max_chars=300,
            height=100
        )
        
        if st.button("🔧 Apply Tweaks") and instruction:
            with st.spinner("Processing instructions..."):
                # Simple tweaking logic for testing
                original_forecast = st.session_state['agent1_forecast'].copy()
                
                # Basic keyword-based adjustments (simplified for testing)
                if "reduce" in instruction.lower() or "decrease" in instruction.lower():
                    if "15%" in instruction:
                        adjustment = 0.85
                    elif "10%" in instruction:
                        adjustment = 0.90
                    else:
                        adjustment = 0.95
                    tweaked_forecast = original_forecast * adjustment
                    explanation = f"Applied reduction factor of {adjustment}"
                    
                elif "increase" in instruction.lower() or "higher" in instruction.lower():
                    if "35%" in instruction:
                        adjustment = 1.35
                    elif "25%" in instruction:
                        adjustment = 1.25
                    else:
                        adjustment = 1.10
                    tweaked_forecast = original_forecast * adjustment
                    explanation = f"Applied increase factor of {adjustment}"
                    
                elif "smooth" in instruction.lower():
                    # Simple smoothing
                    tweaked_forecast = np.convolve(original_forecast, [0.25, 0.5, 0.25], mode='same')
                    explanation = "Applied smoothing filter to reduce volatility"
                    
                else:
                    # Default small adjustment
                    tweaked_forecast = original_forecast * 1.05
                    explanation = "Applied default 5% increase based on instruction"
                
                st.session_state['agent2_forecast'] = tweaked_forecast
                st.session_state['tweak_explanation'] = explanation
                
                st.success("✅ Forecast tweaked successfully!")
        
        # Display tweaked forecast
        if 'agent2_forecast' in st.session_state:
            st.subheader("📊 Tweaked Forecast Results")
            
            original = st.session_state['agent1_forecast']
            tweaked = st.session_state['agent2_forecast']
            
            # Comparison table
            comparison_df = pd.DataFrame({
                'Period': range(1, len(original) + 1),
                'Agent 1 (Original)': original.astype(int),
                'Agent 2 (Tweaked)': tweaked.astype(int),
                'Difference': (tweaked - original).astype(int),
                'Change %': ((tweaked - original) / original * 100).round(1)
            })
            
            st.dataframe(comparison_df)
            
            # Visualization
            fig = go.Figure()
            
            periods = list(range(1, len(original) + 1))
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=original.tolist(),
                mode='lines+markers',
                name='Agent 1 (Original)',
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=tweaked.tolist(),
                mode='lines+markers',
                name='Agent 2 (Tweaked)',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="Forecast Comparison: Agent 1 vs Agent 2",
                xaxis_title="Forecast Period",
                yaxis_title="Call Volume",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            if 'tweak_explanation' in st.session_state:
                st.info(f"**Adjustment Applied:** {st.session_state['tweak_explanation']}")
    
    elif page == "💡 Explanation":
        st.header("💡 Agent 3: Forecast Explanation")
        
        if 'agent1_forecast' not in st.session_state:
            st.warning("⚠️ Please generate a forecast first")
            return
        
        st.info("🤔 Ask questions about the forecasting results")
        
        # Predefined questions
        st.subheader("🔍 Common Questions")
        
        questions = [
            "Why is the forecast higher in some months?",
            "How reliable is this forecast?",
            "What factors influenced the forecast?",
            "Why does Agent 2 differ from Agent 1?",
            "What is the forecast methodology?"
        ]
        
        selected_question = st.selectbox("Select a question:", [""] + questions)
        
        custom_question = st.text_input("Or ask your own question:")
        
        question = custom_question if custom_question else selected_question
        
        if st.button("💡 Get Explanation") and question:
            with st.spinner("Generating explanation..."):
                # Simple explanation logic for testing
                if "higher" in question.lower():
                    explanation = """
                    The forecast shows higher values in certain months due to:
                    1. **Seasonal Patterns**: Historical data shows increased call volumes during specific periods
                    2. **Trend Analysis**: The underlying trend suggests growth over time
                    3. **Driver Correlation**: The driver forecast indicates higher activity in those periods
                    """
                
                elif "reliable" in question.lower():
                    data_points = len(st.session_state['data'])
                    explanation = f"""
                    **Forecast Reliability Assessment:**
                    - **Data Quality**: Based on {data_points} historical data points
                    - **Method**: Simple trend and seasonal analysis
                    - **Confidence**: Moderate (this is a simplified demo version)
                    - **Recommendation**: Use for directional planning, validate with business context
                    """
                
                elif "factors" in question.lower():
                    explanation = """
                    **Key Factors Influencing the Forecast:**
                    1. **Historical Trend**: Long-term growth pattern in call volumes
                    2. **Seasonal Patterns**: Monthly variations based on historical data
                    3. **Driver Forecast**: Correlation with provided driver variables
                    4. **Recent Performance**: Latest actual values influence near-term predictions
                    """
                
                elif "agent 2" in question.lower() or "differ" in question.lower():
                    if 'agent2_forecast' in st.session_state:
                        explanation = f"""
                        **Agent 2 vs Agent 1 Differences:**
                        - **Agent 1**: Pure statistical forecast based on historical patterns
                        - **Agent 2**: Incorporates business logic and human insights
                        - **Adjustment**: {st.session_state.get('tweak_explanation', 'Business-driven modification')}
                        - **Purpose**: Align statistical forecast with business expectations
                        """
                    else:
                        explanation = "Agent 2 forecast not yet generated. Please use the Tweaking section first."
                
                elif "methodology" in question.lower():
                    explanation = """
                    **Forecasting Methodology:**
                    1. **Data Analysis**: Examine historical patterns and trends
                    2. **Statistical Modeling**: Apply trend and seasonal decomposition
                    3. **Driver Integration**: Incorporate driver forecast when significant
                    4. **Business Adjustment**: Apply human insights via Agent 2
                    5. **Validation**: Compare results and assess reasonableness
                    """
                
                else:
                    explanation = f"""
                    **General Forecast Information:**
                    Your question: "{question}"
                    
                    This is a simplified demo version. The full system would provide detailed,
                    data-driven explanations using the actual AI agents and your specific data patterns.
                    
                    Key aspects of the current forecast:
                    - Based on trend and seasonal analysis
                    - Uses {len(st.session_state['data'])} historical data points
                    - Generates {len(st.session_state['agent1_forecast'])} period forecast
                    """
                
                st.markdown("### 🎯 Explanation")
                st.markdown(explanation)
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 Agentic AI Forecasting System** - Standalone Test Version")

if __name__ == "__main__":
    main()
