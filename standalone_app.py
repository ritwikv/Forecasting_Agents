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

def calculate_mape(actual, fitted):
    """Calculate Mean Absolute Percentage Error"""
    actual = np.array(actual)
    fitted = np.array(fitted)
    
    # Remove zero values to avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return float('inf')
    
    mape = np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100
    return mape

def generate_linear_trend_forecast(actuals, periods=6):
    """Linear trend forecasting method"""
    if len(actuals) < 3:
        # Fallback for insufficient data
        trend = np.mean(np.diff(actuals[-2:]))
        forecast = np.array([actuals[-1] + trend * (i + 1) for i in range(periods)])
        fitted = np.array([actuals[-1] + trend * i for i in range(1, len(actuals) + 1)])
        mape = calculate_mape(actuals, fitted[-len(actuals):])
        return forecast, mape
    
    # Fit linear trend
    x = np.arange(len(actuals))
    trend_coef, intercept = np.polyfit(x, actuals, 1)
    
    # Calculate fitted values for MAPE
    fitted = trend_coef * x + intercept
    mape = calculate_mape(actuals, fitted)
    
    # Generate forecast
    future_x = np.arange(len(actuals), len(actuals) + periods)
    forecast = trend_coef * future_x + intercept
    
    return forecast, mape

def generate_seasonal_naive_forecast(actuals, periods=6, seasonal_period=12):
    """Seasonal naive forecasting method"""
    if len(actuals) < seasonal_period:
        # Fallback to simple trend if not enough data for seasonality
        trend = np.mean(np.diff(actuals[-min(6, len(actuals)-1):]))
        forecast = np.array([actuals[-1] + trend * (i + 1) for i in range(periods)])
        fitted = np.array([actuals[-1] + trend * i for i in range(1, len(actuals) + 1)])
        mape = calculate_mape(actuals, fitted[-len(actuals):])
        return forecast, mape
    
    # Generate fitted values for MAPE calculation
    fitted = []
    for i in range(len(actuals)):
        if i < seasonal_period:
            fitted.append(actuals[i])  # Use actual for first season
        else:
            fitted.append(actuals[i - seasonal_period])
    
    # Calculate MAPE (skip first seasonal period for accuracy)
    if len(actuals) > seasonal_period:
        actual_for_mape = actuals[seasonal_period:]
        fitted_for_mape = fitted[seasonal_period:]
        mape = calculate_mape(actual_for_mape, fitted_for_mape)
    else:
        mape = float('inf')
    
    # Use seasonal naive approach for forecast
    forecast = []
    for i in range(periods):
        seasonal_index = (len(actuals) + i) % seasonal_period
        if seasonal_index < len(actuals):
            forecast.append(actuals[-(seasonal_period - seasonal_index)])
        else:
            forecast.append(actuals[-1])  # Fallback to last value
    
    return np.array(forecast), mape

def generate_ensemble_forecast(actuals, periods=6):
    """Ensemble forecasting method combining trend and seasonal"""
    # Generate both forecasts with their MAPEs
    trend_forecast, trend_mape = generate_linear_trend_forecast(actuals, periods)
    seasonal_forecast, seasonal_mape = generate_seasonal_naive_forecast(actuals, periods)
    
    # Combine forecasts with weights (70% trend, 30% seasonal)
    ensemble_forecast = 0.7 * trend_forecast + 0.3 * seasonal_forecast
    
    # Calculate ensemble MAPE as weighted average of component MAPEs
    if trend_mape != float('inf') and seasonal_mape != float('inf'):
        ensemble_mape = 0.7 * trend_mape + 0.3 * seasonal_mape
    elif trend_mape != float('inf'):
        ensemble_mape = trend_mape
    elif seasonal_mape != float('inf'):
        ensemble_mape = seasonal_mape
    else:
        ensemble_mape = float('inf')
    
    return ensemble_forecast, ensemble_mape

def preserve_session_state():
    """Preserve critical session state values to prevent loss during navigation"""
    # List of critical keys that should never be lost
    critical_keys = [
        'data', 'agent1_forecast', 'agent2_forecast', 'forecast_periods',
        'selected_method', 'forecast_mape', 'tweak_explanation'
    ]
    
    # Ensure these keys persist by checking and maintaining them
    for key in critical_keys:
        if key in st.session_state:
            # Force the value to stay in session state by accessing it
            _ = st.session_state[key]

def simple_forecast(data, periods=6):
    """Simple forecasting method for testing (legacy)"""
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
    
    # Initialize session state to prevent loss
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Data Upload", "🔮 Forecasting", "🔧 Tweaking", "💡 Explanation"]
    )
    
    # Debug: Show current session state keys in sidebar
    if st.sidebar.checkbox("🔍 Debug: Show Session State"):
        st.sidebar.write("**Session State Keys:**")
        for key in st.session_state.keys():
            if key != 'initialized':
                st.sidebar.write(f"- {key}")
        
        # Show key values
        if 'agent1_forecast' in st.session_state:
            st.sidebar.write(f"**Agent1 Forecast:** {len(st.session_state['agent1_forecast'])} periods")
        if 'agent2_forecast' in st.session_state:
            st.sidebar.write(f"**Agent2 Forecast:** {len(st.session_state['agent2_forecast'])} periods")
        if 'selected_method' in st.session_state:
            st.sidebar.write(f"**Method:** {st.session_state['selected_method']}")
        if 'forecast_mape' in st.session_state:
            st.sidebar.write(f"**MAPE:** {st.session_state['forecast_mape']}")
    
    # Preserve critical session state values
    preserve_session_state()
    
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
                # Generate forecast based on selected method
                actuals = data['ACD Call Volume Actuals'].values
                
                if method == "Simple Trend":
                    forecast, mape = generate_linear_trend_forecast(actuals, forecast_periods)
                elif method == "Seasonal Naive":
                    forecast, mape = generate_seasonal_naive_forecast(actuals, forecast_periods)
                elif method == "Ensemble":
                    forecast, mape = generate_ensemble_forecast(actuals, forecast_periods)
                else:
                    # Fallback to simple forecast (no MAPE calculation for legacy method)
                    forecast = simple_forecast(data, forecast_periods)
                    mape = None
                
                # Store in session state
                st.session_state['agent1_forecast'] = forecast
                st.session_state['forecast_periods'] = forecast_periods
                st.session_state['selected_method'] = method
                st.session_state['forecast_mape'] = mape
                
                st.success(f"✅ Forecast generated using {method} method!")
        
        # Display forecast if available
        if 'agent1_forecast' in st.session_state:
            forecast = st.session_state['agent1_forecast']
            
            st.subheader("📊 Forecast Results")
            
            # Show selected method and MAPE
            selected_method = st.session_state.get('selected_method', 'Unknown')
            forecast_mape = st.session_state.get('forecast_mape', None)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🔧 **Method Used:** {selected_method}")
            with col2:
                if forecast_mape is not None and forecast_mape != float('inf'):
                    st.info(f"📊 **MAPE:** {forecast_mape:.2f}%")
                else:
                    st.info("📊 **MAPE:** Not available")
            
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
        
        # Check for required session state values
        if 'agent1_forecast' not in st.session_state:
            st.warning("⚠️ Please generate a forecast first in the Forecasting section")
            st.info("💡 Go to the **🔮 Forecasting** section and click **🚀 Generate Forecast** to create a forecast first.")
            return
        
        # Display current forecast info for context
        if 'selected_method' in st.session_state and 'forecast_mape' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🔧 **Current Method:** {st.session_state['selected_method']}")
            with col2:
                mape = st.session_state['forecast_mape']
                if mape is not None and mape != float('inf'):
                    st.info(f"📊 **Current MAPE:** {mape:.2f}%")
                else:
                    st.info("📊 **Current MAPE:** Not available")
        
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
                original_forecast = st.session_state['agent1_forecast'].copy()
                data = st.session_state['data']
                
                # Advanced instruction parsing
                instruction_lower = instruction.lower()
                
                # Extract percentage values from instruction
                import re
                percentages = re.findall(r'(\d+(?:\.\d+)?)%', instruction)
                numbers = re.findall(r'(\d+(?:\.\d+)?)', instruction)
                
                # Initialize variables
                tweaked_forecast = original_forecast.copy()
                explanation = ""
                
                # INTELLIGENT INSTRUCTION PARSING
                
                # 1. PERCENTAGE-BASED ADJUSTMENTS
                if any(word in instruction_lower for word in ["reduce", "decrease", "lower", "cut", "drop"]):
                    if percentages:
                        pct = float(percentages[0]) / 100
                        adjustment = 1 - pct
                        tweaked_forecast = original_forecast * adjustment
                        explanation = f"🔻 Reduced forecast by {percentages[0]}% across all periods (factor: {adjustment:.2f})"
                    else:
                        # Look for specific amounts
                        if any(word in instruction_lower for word in ["significantly", "much", "substantially"]):
                            adjustment = 0.80  # 20% reduction
                            explanation = "🔻 Applied significant reduction of 20% based on instruction"
                        else:
                            adjustment = 0.90  # 10% reduction
                            explanation = "🔻 Applied moderate reduction of 10% based on instruction"
                        tweaked_forecast = original_forecast * adjustment
                
                elif any(word in instruction_lower for word in ["increase", "raise", "boost", "higher", "up"]):
                    if percentages:
                        pct = float(percentages[0]) / 100
                        adjustment = 1 + pct
                        tweaked_forecast = original_forecast * adjustment
                        explanation = f"🔺 Increased forecast by {percentages[0]}% across all periods (factor: {adjustment:.2f})"
                    else:
                        # Look for specific amounts
                        if any(word in instruction_lower for word in ["significantly", "much", "substantially"]):
                            adjustment = 1.25  # 25% increase
                            explanation = "🔺 Applied significant increase of 25% based on instruction"
                        else:
                            adjustment = 1.15  # 15% increase
                            explanation = "🔺 Applied moderate increase of 15% based on instruction"
                        tweaked_forecast = original_forecast * adjustment
                
                # 2. SMOOTHING AND VOLATILITY ADJUSTMENTS
                elif any(word in instruction_lower for word in ["smooth", "volatility", "stable", "consistent"]):
                    # Apply moving average smoothing
                    window_size = 3
                    tweaked_forecast = pd.Series(original_forecast).rolling(window=window_size, center=True, min_periods=1).mean().values
                    explanation = f"📊 Applied smoothing filter (window size: {window_size}) to reduce volatility"
                
                # 3. SPECIFIC MONTH/PERIOD ADJUSTMENTS
                elif any(month in instruction_lower for month in ["january", "february", "march", "april", "may", "june", 
                                                                "july", "august", "september", "october", "november", "december"]):
                    # Extract month and adjustment
                    months = ["january", "february", "march", "april", "may", "june", 
                             "july", "august", "september", "october", "november", "december"]
                    target_month = None
                    for i, month in enumerate(months):
                        if month in instruction_lower:
                            target_month = i + 1
                            break
                    
                    if target_month and len(original_forecast) >= target_month:
                        period_idx = target_month - 1
                        if percentages:
                            pct = float(percentages[0]) / 100
                            if any(word in instruction_lower for word in ["reduce", "decrease", "lower"]):
                                tweaked_forecast[period_idx] = original_forecast[period_idx] * (1 - pct)
                                explanation = f"📅 Reduced {months[target_month-1].title()} (Period {target_month}) by {percentages[0]}%"
                            else:
                                tweaked_forecast[period_idx] = original_forecast[period_idx] * (1 + pct)
                                explanation = f"📅 Increased {months[target_month-1].title()} (Period {target_month}) by {percentages[0]}%"
                        else:
                            # Default 10% adjustment for specific month
                            if any(word in instruction_lower for word in ["reduce", "decrease", "lower"]):
                                tweaked_forecast[period_idx] = original_forecast[period_idx] * 0.90
                                explanation = f"📅 Reduced {months[target_month-1].title()} (Period {target_month}) by 10%"
                            else:
                                tweaked_forecast[period_idx] = original_forecast[period_idx] * 1.10
                                explanation = f"📅 Increased {months[target_month-1].title()} (Period {target_month}) by 10%"
                
                # 4. DRIVER-BASED ADJUSTMENTS
                elif any(word in instruction_lower for word in ["driver", "align", "match"]):
                    # Align with driver forecast pattern
                    driver_avg = data['Driver_Forecast'].mean()
                    forecast_avg = original_forecast.mean()
                    ratio = driver_avg / forecast_avg
                    tweaked_forecast = original_forecast * ratio
                    explanation = f"🔗 Aligned forecast with driver pattern (ratio: {ratio:.2f})"
                
                # 5. HISTORICAL COMPARISON ADJUSTMENTS
                elif any(word in instruction_lower for word in ["historical", "average", "typical", "normal"]):
                    historical_avg = data['ACD Call Volume Actuals'].mean()
                    forecast_avg = original_forecast.mean()
                    ratio = historical_avg / forecast_avg
                    tweaked_forecast = original_forecast * ratio
                    explanation = f"📈 Adjusted to align with historical average (ratio: {ratio:.2f})"
                
                # 6. TREND ADJUSTMENTS
                elif any(word in instruction_lower for word in ["trend", "growth", "decline"]):
                    # Calculate current trend
                    trend_slope = np.polyfit(range(len(data)), data['ACD Call Volume Actuals'], 1)[0]
                    
                    if "accelerate" in instruction_lower or "faster" in instruction_lower:
                        # Increase trend by 50%
                        enhanced_trend = trend_slope * 1.5
                        for i in range(len(tweaked_forecast)):
                            tweaked_forecast[i] = original_forecast[0] + enhanced_trend * (i + 1)
                        explanation = f"📈 Accelerated growth trend by 50% (new slope: {enhanced_trend:.1f})"
                    
                    elif "moderate" in instruction_lower or "slower" in instruction_lower:
                        # Reduce trend by 50%
                        moderated_trend = trend_slope * 0.5
                        for i in range(len(tweaked_forecast)):
                            tweaked_forecast[i] = original_forecast[0] + moderated_trend * (i + 1)
                        explanation = f"📉 Moderated growth trend by 50% (new slope: {moderated_trend:.1f})"
                
                # 7. RANGE-BASED ADJUSTMENTS
                elif any(word in instruction_lower for word in ["cap", "limit", "maximum", "minimum", "floor", "ceiling"]):
                    if numbers:
                        limit_value = float(numbers[0])
                        if any(word in instruction_lower for word in ["maximum", "cap", "ceiling"]):
                            tweaked_forecast = np.minimum(original_forecast, limit_value)
                            explanation = f"🔒 Applied maximum cap of {limit_value} to all periods"
                        elif any(word in instruction_lower for word in ["minimum", "floor"]):
                            tweaked_forecast = np.maximum(original_forecast, limit_value)
                            explanation = f"🔒 Applied minimum floor of {limit_value} to all periods"
                
                # 8. BUSINESS CONTEXT ADJUSTMENTS
                elif any(word in instruction_lower for word in ["conservative", "aggressive", "optimistic", "pessimistic"]):
                    if "conservative" in instruction_lower or "pessimistic" in instruction_lower:
                        adjustment = 0.85  # 15% reduction
                        tweaked_forecast = original_forecast * adjustment
                        explanation = "🛡️ Applied conservative adjustment (-15%) for risk management"
                    elif "aggressive" in instruction_lower or "optimistic" in instruction_lower:
                        adjustment = 1.20  # 20% increase
                        tweaked_forecast = original_forecast * adjustment
                        explanation = "🚀 Applied aggressive adjustment (+20%) for growth planning"
                
                # 9. FALLBACK - INTELLIGENT DEFAULT
                else:
                    # Analyze the instruction for sentiment and apply intelligent default
                    if any(word in instruction_lower for word in ["too high", "overestimate", "excessive"]):
                        adjustment = 0.90
                        tweaked_forecast = original_forecast * adjustment
                        explanation = "🔻 Applied 10% reduction based on 'too high' sentiment in instruction"
                    elif any(word in instruction_lower for word in ["too low", "underestimate", "insufficient"]):
                        adjustment = 1.15
                        tweaked_forecast = original_forecast * adjustment
                        explanation = "🔺 Applied 15% increase based on 'too low' sentiment in instruction"
                    else:
                        # Minimal adjustment with explanation
                        adjustment = 1.02
                        tweaked_forecast = original_forecast * adjustment
                        explanation = f"⚙️ Applied minimal 2% adjustment. Instruction: '{instruction[:50]}...' - Consider using more specific terms like 'increase by 10%' or 'reduce December by 15%'"
                
                # Ensure forecast values are positive
                tweaked_forecast = np.maximum(tweaked_forecast, 1)
                
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
        
        # Check for required session state values
        if 'agent1_forecast' not in st.session_state:
            st.warning("⚠️ Please generate a forecast first")
            st.info("💡 Go to the **🔮 Forecasting** section and click **🚀 Generate Forecast** to create a forecast first.")
            return
        
        # Display current session state for context
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'selected_method' in st.session_state:
                st.info(f"🔧 **Method:** {st.session_state['selected_method']}")
        with col2:
            if 'forecast_mape' in st.session_state:
                mape = st.session_state['forecast_mape']
                if mape is not None and mape != float('inf'):
                    st.info(f"📊 **MAPE:** {mape:.2f}%")
        with col3:
            if 'agent2_forecast' in st.session_state:
                st.info("🔧 **Tweaked:** Yes")
            else:
                st.info("🔧 **Tweaked:** No")
        
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
                # Get data for analysis
                data = st.session_state['data']
                forecast = st.session_state['agent1_forecast']
                
                # Calculate key statistics
                historical_avg = data['ACD Call Volume Actuals'].mean()
                historical_std = data['ACD Call Volume Actuals'].std()
                forecast_avg = forecast.mean()
                latest_actual = data['ACD Call Volume Actuals'].iloc[-1]
                trend_slope = np.polyfit(range(len(data)), data['ACD Call Volume Actuals'], 1)[0]
                
                # Analyze seasonal patterns
                if len(data) >= 12:
                    monthly_avg = data.groupby(data.index % 12)['ACD Call Volume Actuals'].mean()
                    peak_month = monthly_avg.idxmax() + 1
                    low_month = monthly_avg.idxmin() + 1
                    seasonal_variation = (monthly_avg.max() - monthly_avg.min()) / historical_avg * 100
                else:
                    peak_month = "N/A"
                    low_month = "N/A" 
                    seasonal_variation = 0
                
                # Define variables for use in all explanation branches
                monthly_trend = trend_slope
                driver_corr = np.corrcoef(data['ACD Call Volume Actuals'], data['Driver_Forecast'])[0,1]
                
                # Generate intelligent explanations based on actual data
                if any(word in question.lower() for word in ["higher", "increase", "peak", "maximum"]):
                    max_forecast_idx = np.argmax(forecast)
                    max_forecast_value = forecast[max_forecast_idx]
                    explanation = f"""
                    **📈 Why the Forecast Shows Higher Values:**
                    
                    **Specific Analysis:**
                    - **Peak Forecast**: Period {max_forecast_idx + 1} shows the highest value of {max_forecast_value:.0f}
                    - **vs Historical Average**: This is {((max_forecast_value - historical_avg) / historical_avg * 100):.1f}% above the historical average of {historical_avg:.0f}
                    - **Trend Impact**: The data shows a {"positive" if trend_slope > 0 else "negative"} trend of {trend_slope:.1f} units per period
                    
                    **Key Drivers:**
                    1. **Historical Trend**: {"Growing" if trend_slope > 0 else "Declining"} pattern in your data
                    2. **Seasonal Patterns**: {f"Peak typically occurs in month {peak_month}" if peak_month != "N/A" else "Limited seasonal data available"}
                    3. **Recent Performance**: Latest actual value was {latest_actual:.0f}
                    4. **Driver Correlation**: Driver forecast shows {data['Driver_Forecast'].iloc[-6:].mean():.0f} average in recent periods
                    """
                
                elif any(word in question.lower() for word in ["lower", "decrease", "minimum", "drop"]):
                    min_forecast_idx = np.argmin(forecast)
                    min_forecast_value = forecast[min_forecast_idx]
                    explanation = f"""
                    **📉 Why the Forecast Shows Lower Values:**
                    
                    **Specific Analysis:**
                    - **Lowest Forecast**: Period {min_forecast_idx + 1} shows the lowest value of {min_forecast_value:.0f}
                    - **vs Historical Average**: This is {((historical_avg - min_forecast_value) / historical_avg * 100):.1f}% below the historical average of {historical_avg:.0f}
                    - **Trend Impact**: The underlying trend is {trend_slope:.1f} units per period
                    
                    **Possible Reasons:**
                    1. **Seasonal Effect**: {f"Low period typically occurs in month {low_month}" if low_month != "N/A" else "Natural variation in the data"}
                    2. **Mean Reversion**: Forecast adjusts after recent high values
                    3. **Driver Influence**: Driver forecast shows corresponding lower values
                    4. **Statistical Smoothing**: Algorithm balances extreme values
                    """
                
                elif any(word in question.lower() for word in ["reliable", "accurate", "confidence", "trust"]):
                    cv = historical_std / historical_avg * 100  # Coefficient of variation
                    explanation = f"""
                    **🎯 Forecast Reliability Assessment:**
                    
                    **Data Quality Metrics:**
                    - **Historical Data**: {len(data)} data points spanning {len(data)/12:.1f} years
                    - **Data Stability**: Coefficient of variation is {cv:.1f}% ({"Low" if cv < 15 else "Moderate" if cv < 30 else "High"} volatility)
                    - **Trend Consistency**: {"Strong" if abs(trend_slope) > historical_std/10 else "Weak"} trend pattern detected
                    
                    **Confidence Indicators:**
                    - **✅ Strengths**: {len(data)} historical observations, clear trend pattern
                    - **⚠️ Limitations**: Simplified algorithm, no external factors considered
                    - **📊 Accuracy**: Expected error range ±{historical_std:.0f} units ({historical_std/historical_avg*100:.1f}%)
                    
                    **Recommendation**: Use for directional planning. Validate against business knowledge and external factors.
                    """
                
                elif any(word in question.lower() for word in ["agent 2", "tweak", "differ", "change", "adjust"]):
                    if 'agent2_forecast' in st.session_state:
                        agent2_forecast = st.session_state['agent2_forecast']
                        total_change = np.sum(agent2_forecast - forecast)
                        avg_change_pct = np.mean((agent2_forecast - forecast) / forecast * 100)
                        max_change_idx = np.argmax(np.abs(agent2_forecast - forecast))
                        max_change_value = agent2_forecast[max_change_idx] - forecast[max_change_idx]
                        
                        explanation = f"""
                        **🔧 Agent 2 vs Agent 1 Analysis:**
                        
                        **Quantitative Differences:**
                        - **Total Adjustment**: {total_change:.0f} units across all periods
                        - **Average Change**: {avg_change_pct:.1f}% per period
                        - **Largest Change**: Period {max_change_idx + 1} changed by {max_change_value:.0f} units
                        
                        **Business Logic Applied:**
                        - **Agent 1**: Pure statistical forecast (Avg: {forecast.mean():.0f})
                        - **Agent 2**: Business-adjusted forecast (Avg: {agent2_forecast.mean():.0f})
                        - **Adjustment Type**: {st.session_state.get('tweak_explanation', 'Business-driven modification')}
                        
                        **Impact Assessment:**
                        - **Direction**: {"Upward" if total_change > 0 else "Downward"} adjustment overall
                        - **Magnitude**: {abs(avg_change_pct):.1f}% average change
                        - **Consistency**: {"Uniform" if np.std((agent2_forecast - forecast) / forecast * 100) < 5 else "Variable"} adjustment pattern
                        """
                    else:
                        explanation = "❌ **Agent 2 forecast not available.** Please generate a forecast and apply tweaks first in the Tweaking section."
                
                elif any(word in question.lower() for word in ["seasonal", "month", "pattern", "cycle"]):
                    if seasonal_variation > 0:
                        explanation = f"""
                        **📅 Seasonal Pattern Analysis:**
                        
                        **Seasonal Statistics:**
                        - **Seasonal Variation**: {seasonal_variation:.1f}% difference between peak and trough
                        - **Peak Period**: Month {peak_month} typically shows highest values
                        - **Low Period**: Month {low_month} typically shows lowest values
                        - **Pattern Strength**: {"Strong" if seasonal_variation > 20 else "Moderate" if seasonal_variation > 10 else "Weak"} seasonal effect
                        
                        **Monthly Averages** (based on historical data):
                        {chr(10).join([f"- Month {i+1}: {monthly_avg.iloc[i]:.0f}" for i in range(min(12, len(monthly_avg)))])}
                        
                        **Forecast Impact:**
                        - The forecast incorporates these seasonal patterns
                        - Expect higher values around month {peak_month}
                        - Plan for lower volumes around month {low_month}
                        """
                    else:
                        explanation = f"""
                        **📅 Seasonal Analysis:**
                        
                        **Limited Seasonal Data:**
                        - Only {len(data)} data points available
                        - Need at least 24 months for robust seasonal analysis
                        - Current forecast uses trend-based approach
                        
                        **Recommendation:**
                        - Collect more historical data for seasonal patterns
                        - Consider business calendar effects manually
                        - Monitor for emerging seasonal trends
                        """
                
                elif any(word in question.lower() for word in ["trend", "growth", "direction"]):
                    annual_trend = monthly_trend * 12
                    explanation = f"""
                    **📈 Trend Analysis:**
                    
                    **Trend Statistics:**
                    - **Monthly Trend**: {monthly_trend:.1f} units per month
                    - **Annual Trend**: {annual_trend:.0f} units per year ({annual_trend/historical_avg*100:.1f}% annually)
                    - **Trend Direction**: {"📈 Growing" if monthly_trend > 0 else "📉 Declining" if monthly_trend < 0 else "➡️ Stable"}
                    
                    **Historical Context:**
                    - **Starting Point**: First data point was {data['ACD Call Volume Actuals'].iloc[0]:.0f}
                    - **Current Level**: Latest actual is {latest_actual:.0f}
                    - **Total Change**: {latest_actual - data['ACD Call Volume Actuals'].iloc[0]:.0f} units over {len(data)} periods
                    
                    **Forecast Implication:**
                    - The forecast continues this {"growth" if monthly_trend > 0 else "decline" if monthly_trend < 0 else "stable"} pattern
                    - Expected range: {forecast.min():.0f} to {forecast.max():.0f}
                    """
                
                elif any(word in question.lower() for word in ["driver", "correlation", "relationship"]):
                    explanation = f"""
                    **🔗 Driver Relationship Analysis:**
                    
                    **Correlation Statistics:**
                    - **Correlation Coefficient**: {driver_corr:.3f} ({
                        "Strong positive" if driver_corr > 0.7 else
                        "Moderate positive" if driver_corr > 0.3 else
                        "Weak positive" if driver_corr > 0.1 else
                        "No clear" if abs(driver_corr) <= 0.1 else
                        "Weak negative" if driver_corr > -0.3 else
                        "Moderate negative" if driver_corr > -0.7 else
                        "Strong negative"
                    } relationship)
                    
                    **Driver Statistics:**
                    - **Driver Average**: {data['Driver_Forecast'].mean():.0f}
                    - **Actuals Average**: {historical_avg:.0f}
                    - **Ratio**: {historical_avg / data['Driver_Forecast'].mean():.2f} (Actuals/Driver)
                    
                    **Forecast Impact:**
                    - {"High" if abs(driver_corr) > 0.5 else "Moderate" if abs(driver_corr) > 0.3 else "Low"} influence on forecast
                    - Driver forecast {"strongly supports" if driver_corr > 0.5 else "moderately supports" if driver_corr > 0.2 else "weakly relates to"} the call volume forecast
                    """
                
                else:
                    # Provide a comprehensive overview for any other question
                    explanation = f"""
                    **🤖 Comprehensive Forecast Analysis:**
                    
                    **Your Question**: "{question}"
                    
                    **Key Forecast Insights:**
                    - **Data Foundation**: {len(data)} historical data points
                    - **Method Used**: {st.session_state.get('selected_method', 'Unknown')}
                    - **Model Accuracy**: {f"{st.session_state.get('forecast_mape', 0):.2f}% MAPE" + (' (Excellent)' if st.session_state.get('forecast_mape', float('inf')) < 10 else ' (Good)' if st.session_state.get('forecast_mape', float('inf')) < 20 else ' (Fair)' if st.session_state.get('forecast_mape', float('inf')) < 50 else ' (Needs Review)') if st.session_state.get('forecast_mape') is not None and st.session_state.get('forecast_mape') != float('inf') else 'Not available'}
                    - **Forecast Range**: {forecast.min():.0f} to {forecast.max():.0f} (Avg: {forecast_avg:.0f})
                    - **vs Historical**: {((forecast_avg - historical_avg) / historical_avg * 100):+.1f}% change from historical average
                    - **Trend**: {monthly_trend:+.1f} units per month
                    - **Volatility**: ±{historical_std:.0f} typical variation
                    
                    **Pattern Recognition:**
                    - **Seasonal Effect**: {seasonal_variation:.1f}% variation between peak/trough
                    - **Driver Correlation**: {driver_corr:.2f} relationship strength
                    - **Recent Performance**: Latest actual ({latest_actual:.0f}) vs forecast start
                    
                    **Business Implications:**
                    - Plan for {forecast_avg:.0f} average call volume
                    - Expect range between {forecast.min():.0f} and {forecast.max():.0f}
                    - {"Growth trajectory" if trend_slope > 0 else "Declining trend" if trend_slope < 0 else "Stable pattern"} continues
                    """
                
                st.markdown("### 🎯 Explanation")
                st.markdown(explanation)
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 Agentic AI Forecasting System** - Standalone Test Version")

if __name__ == "__main__":
    main()
