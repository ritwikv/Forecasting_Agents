"""
Main Streamlit application for the Agentic AI Forecasting System
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import traceback

# Import components
from .components.data_upload import DataUploadComponent
from .components.forecast_display import ForecastDisplayComponent
from .components.tweaking_interface import TweakingInterfaceComponent
from .components.explanation_interface import ExplanationInterfaceComponent
from .utils.chart_generator import ChartGenerator

# Import system components
from agents.agent_orchestrator import AgentOrchestrator
from config import STREAMLIT_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingApp:
    """
    Main Streamlit application class
    """
    
    def __init__(self):
        """Initialize the application"""
        self.setup_page_config()
        self.initialize_session_state()
        self.initialize_components()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=STREAMLIT_CONFIG["page_title"],
            page_icon=STREAMLIT_CONFIG["page_icon"],
            layout=STREAMLIT_CONFIG["layout"],
            initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'agent1_completed' not in st.session_state:
            st.session_state.agent1_completed = False
        
        if 'agent2_completed' not in st.session_state:
            st.session_state.agent2_completed = False
        
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'data_loading'
        
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = {}
        
        if 'model_path' not in st.session_state:
            st.session_state.model_path = MODEL_CONFIG["model_path"]
    
    def initialize_components(self):
        """Initialize UI components"""
        self.data_upload = DataUploadComponent()
        self.forecast_display = ForecastDisplayComponent()
        self.tweaking_interface = TweakingInterfaceComponent()
        self.explanation_interface = ExplanationInterfaceComponent()
        self.chart_generator = ChartGenerator()
    
    def run(self):
        """Run the main application"""
        try:
            # Header
            st.title("🤖 Agentic AI Forecasting System")
            st.markdown("---")
            
            # Sidebar
            self.render_sidebar()
            
            # Main content
            self.render_main_content()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {str(e)}")
            logger.error(traceback.format_exc())
    
    def render_sidebar(self):
        """Render sidebar with navigation and status"""
        with st.sidebar:
            st.header("🎯 System Status")
            
            # Model configuration
            st.subheader("Model Configuration")
            model_path = st.text_input(
                "Model Path",
                value=st.session_state.model_path,
                help="Path to Phi-3-mini-4k-instruct-Q4_K_M.gguf file"
            )
            
            if model_path != st.session_state.model_path:
                st.session_state.model_path = model_path
                # Reset orchestrator if model path changes
                st.session_state.orchestrator = None
            
            # Initialize orchestrator button
            if st.button("🚀 Initialize System", type="primary"):
                self.initialize_orchestrator()
            
            # System status
            st.subheader("System Status")
            if st.session_state.orchestrator:
                status = st.session_state.orchestrator.get_workflow_status()
                
                st.success("✅ System Initialized")
                st.info(f"Current Step: {status['current_step']}")
                
                # Progress bar
                progress = status['progress']['progress_percentage'] / 100
                st.progress(progress)
                st.caption(f"Progress: {status['progress']['progress_percentage']:.0f}%")
                
                # Workflow steps
                st.subheader("Workflow Steps")
                for i, step in enumerate(status['progress']['all_steps']):
                    if step in status['progress']['completed_steps']:
                        st.success(f"✅ {step.replace('_', ' ').title()}")
                    elif step == status['current_step']:
                        st.info(f"🔄 {step.replace('_', ' ').title()}")
                    else:
                        st.write(f"⏳ {step.replace('_', ' ').title()}")
            else:
                st.warning("⚠️ System Not Initialized")
            
            # Reset button
            if st.button("🔄 Reset Session"):
                self.reset_session()
    
    def render_main_content(self):
        """Render main content area"""
        if not st.session_state.orchestrator:
            self.render_welcome_screen()
        else:
            current_step = st.session_state.orchestrator.workflow_manager.get_current_step()
            
            if current_step == 'data_loading':
                self.render_data_loading_step()
            elif current_step == 'agent1_forecasting':
                self.render_agent1_step()
            elif current_step == 'agent2_tweaking':
                self.render_agent2_step()
            elif current_step == 'agent3_explanation':
                self.render_agent3_step()
            elif current_step == 'results_display':
                self.render_results_step()
            else:
                st.error(f"Unknown workflow step: {current_step}")
    
    def render_welcome_screen(self):
        """Render welcome screen"""
        st.header("Welcome to the Agentic AI Forecasting System")
        
        st.markdown("""
        This system uses three specialized AI agents to provide accurate and explainable forecasts:
        
        ### 🎯 **Agent 1: Forecasting Agent**
        - Forecasts ACD Call Volume using statistical methods
        - Minimizes Mean Absolute Percentage Error (MAPE)
        - Tests Driver_Forecast significance automatically
        - Recommends optimal forecast horizons
        
        ### 🔧 **Agent 2: Tweaking Agent**
        - Processes human business insights (up to 300 words)
        - Adjusts forecasts based on domain expertise
        - Handles various comparison scenarios
        - Maintains forecast integrity
        
        ### 💡 **Agent 3: Explanation Agent**
        - Provides conversational explanations
        - Answers specific questions about forecasts
        - Compares different forecasting approaches
        - Makes complex concepts accessible
        
        ### 🚀 **Getting Started**
        1. Configure your model path in the sidebar
        2. Click "Initialize System" to start
        3. Upload your Excel data file
        4. Follow the guided workflow
        """)
        
        # System requirements
        with st.expander("📋 System Requirements"):
            st.markdown("""
            - **Model**: Phi-3-mini-4k-instruct-Q4_K_M.gguf
            - **RAM**: 16GB recommended
            - **CPU**: Multi-core processor (optimized for CPU inference)
            - **Data**: Excel file with 'ACD Call Volume Actuals' and 'Driver_Forecast' columns
            - **OS**: Windows 11 Pro (optimized for)
            """)
    
    def render_data_loading_step(self):
        """Render data loading step"""
        st.header("📊 Step 1: Data Loading")
        
        # Data upload component
        upload_result = self.data_upload.render()
        
        if upload_result and upload_result.get('success'):
            # Load data using orchestrator
            with st.spinner("Processing data..."):
                result = st.session_state.orchestrator.load_data(upload_result['file_path'])
            
            if result['success']:
                st.success("✅ Data loaded successfully!")
                st.session_state.data_loaded = True
                
                # Display data summary
                st.subheader("Data Summary")
                summary = result['data_summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", summary['total_records'])
                with col2:
                    st.metric("Date Range", f"{summary['date_range']['start'].split()[0]} to {summary['date_range']['end'].split()[0]}")
                with col3:
                    correlation = result['forecast_data']['statistics']['correlation']
                    st.metric("Actuals-Driver Correlation", f"{correlation:.3f}")
                
                # Display data preview
                st.subheader("Data Preview")
                forecast_data = result['forecast_data']
                
                preview_df = pd.DataFrame({
                    'ACD Call Volume Actuals': forecast_data['actuals'][:10],
                    'Driver_Forecast': forecast_data['driver_forecast'][:10]
                })
                st.dataframe(preview_df)
                
                # Next step button
                if st.button("➡️ Proceed to Forecasting", type="primary"):
                    st.session_state.orchestrator.workflow_manager.advance_step()
                    st.rerun()
            else:
                st.error(f"❌ Data loading failed: {result['error']}")
    
    def render_agent1_step(self):
        """Render Agent 1 forecasting step"""
        st.header("🎯 Step 2: Agent 1 - Forecasting")
        
        st.markdown("""
        Agent 1 will analyze your historical data and generate forecasts using statistical methods.
        It will automatically test the significance of Driver_Forecast and select the best model to minimize MAPE.
        """)
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_periods = st.number_input(
                "Forecast Periods",
                min_value=1,
                max_value=24,
                value=12,
                help="Number of periods to forecast (leave blank for auto-recommendation)"
            )
        
        with col2:
            use_target_date = st.checkbox("Specify Target Date")
        
        target_year = None
        target_month = None
        
        if use_target_date:
            col3, col4 = st.columns(2)
            with col3:
                target_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=2025)
            with col4:
                target_month = st.selectbox("Target Month", range(1, 13), index=11)
        
        # Run forecasting
        if st.button("🚀 Run Agent 1 Forecasting", type="primary"):
            with st.spinner("Agent 1 is analyzing data and generating forecasts..."):
                result = st.session_state.orchestrator.run_agent1_forecast(
                    forecast_periods=forecast_periods if not use_target_date else None,
                    target_year=target_year,
                    target_month=target_month
                )
            
            if result['success']:
                st.success("✅ Agent 1 forecasting completed!")
                st.session_state.agent1_completed = True
                st.session_state.forecast_results['agent1'] = result
                
                # Display results
                self.display_agent1_results(result)
                
                # Next step button
                if st.button("➡️ Proceed to Tweaking (Optional)", type="primary"):
                    st.session_state.orchestrator.workflow_manager.advance_step()
                    st.rerun()
            else:
                st.error(f"❌ Agent 1 forecasting failed: {result['error']}")
    
    def render_agent2_step(self):
        """Render Agent 2 tweaking step"""
        st.header("🔧 Step 3: Agent 2 - Forecast Tweaking (Optional)")
        
        st.markdown("""
        Agent 2 can adjust the forecast based on your business insights and domain expertise.
        Provide instructions in natural language (up to 300 words) to modify specific aspects of the forecast.
        """)
        
        # Display current forecast
        if 'agent1' in st.session_state.forecast_results:
            st.subheader("Current Forecast (Agent 1)")
            agent1_result = st.session_state.forecast_results['agent1']
            
            # Create forecast chart
            chart_data = pd.DataFrame({
                'Period': range(1, len(agent1_result['forecast_values']) + 1),
                'Agent 1 Forecast': agent1_result['forecast_values']
            })
            
            st.line_chart(chart_data.set_index('Period'))
            
            # Tweaking interface
            tweak_result = self.tweaking_interface.render(agent1_result)
            
            if tweak_result and tweak_result.get('success'):
                st.session_state.agent2_completed = True
                st.session_state.forecast_results['agent2'] = tweak_result
                
                # Display comparison
                self.display_forecast_comparison(agent1_result, tweak_result)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏭️ Skip Tweaking"):
                st.session_state.orchestrator.workflow_manager.advance_step()
                st.rerun()
        
        with col2:
            if st.session_state.agent2_completed:
                if st.button("➡️ Proceed to Explanation", type="primary"):
                    st.session_state.orchestrator.workflow_manager.advance_step()
                    st.rerun()
    
    def render_agent3_step(self):
        """Render Agent 3 explanation step"""
        st.header("💡 Step 4: Agent 3 - Forecast Explanation")
        
        st.markdown("""
        Agent 3 can provide detailed explanations about the forecasting process and results.
        Ask specific questions or request general explanations about the forecasts.
        """)
        
        # Explanation interface
        explanation_result = self.explanation_interface.render(
            st.session_state.forecast_results
        )
        
        if explanation_result:
            st.session_state.forecast_results['agent3'] = explanation_result
        
        # Next step button
        if st.button("➡️ View Final Results", type="primary"):
            st.session_state.orchestrator.workflow_manager.advance_step()
            st.rerun()
    
    def render_results_step(self):
        """Render final results step"""
        st.header("📈 Final Results")
        
        # Get all results
        all_results = st.session_state.orchestrator.get_all_results()
        
        # Display comprehensive results
        self.forecast_display.render_comprehensive_results(all_results)
        
        # Export options
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export to JSON"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"forecast_results_{timestamp}.json"
                export_result = st.session_state.orchestrator.export_results(filename)
                
                if export_result['success']:
                    st.success(f"✅ Results exported to {filename}")
                else:
                    st.error(f"❌ Export failed: {export_result['error']}")
        
        with col2:
            if st.button("🔄 Start New Session"):
                self.reset_session()
                st.rerun()
    
    def display_agent1_results(self, result: Dict[str, Any]):
        """Display Agent 1 results"""
        st.subheader("Agent 1 Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Method Used", result['method_used'])
        with col2:
            st.metric("MAPE", f"{result['accuracy_metrics']['mape']:.2f}%")
        with col3:
            st.metric("Correlation", f"{result['accuracy_metrics']['correlation']:.3f}")
        with col4:
            driver_sig = "Yes" if result['driver_significance']['is_significant'] else "No"
            st.metric("Driver Significant", driver_sig)
        
        # Forecast chart
        st.subheader("Forecast Chart")
        forecast_df = pd.DataFrame({
            'Period': range(1, len(result['forecast_values']) + 1),
            'Forecast': result['forecast_values']
        })
        st.line_chart(forecast_df.set_index('Period'))
        
        # Forecast table
        st.subheader("Forecast Values")
        st.dataframe(forecast_df)
    
    def display_forecast_comparison(self, agent1_result: Dict[str, Any], agent2_result: Dict[str, Any]):
        """Display comparison between Agent 1 and Agent 2 forecasts"""
        st.subheader("Forecast Comparison")
        
        # Create comparison chart
        comparison_df = pd.DataFrame({
            'Period': range(1, len(agent1_result['forecast_values']) + 1),
            'Agent 1 (Original)': agent1_result['forecast_values'],
            'Agent 2 (Tweaked)': agent2_result['tweaked_forecast']
        })
        
        st.line_chart(comparison_df.set_index('Period'))
        
        # Comparison table
        st.subheader("Comparison Table")
        comparison_df['Difference'] = comparison_df['Agent 2 (Tweaked)'] - comparison_df['Agent 1 (Original)']
        comparison_df['% Change'] = (comparison_df['Difference'] / comparison_df['Agent 1 (Original)'] * 100).round(2)
        
        st.dataframe(comparison_df)
        
        # Impact metrics
        if 'impact_metrics' in agent2_result:
            impact = agent2_result['impact_metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Periods Changed", impact.get('periods_changed', 0))
            with col2:
                st.metric("Avg % Change", f"{impact.get('average_percentage_change', 0):.2f}%")
            with col3:
                st.metric("Net Change", f"{impact.get('net_percentage_change', 0):+.2f}%")
    
    def initialize_orchestrator(self):
        """Initialize the agent orchestrator"""
        try:
            with st.spinner("Initializing AI agents..."):
                orchestrator = AgentOrchestrator(model_path=st.session_state.model_path)
                
                if orchestrator.initialize_agents():
                    st.session_state.orchestrator = orchestrator
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize system. Please check your model path.")
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")
            logger.error(f"Orchestrator initialization error: {str(e)}")
    
    def reset_session(self):
        """Reset the entire session"""
        if st.session_state.orchestrator:
            st.session_state.orchestrator.reset_session()
        
        # Reset session state
        for key in list(st.session_state.keys()):
            if key != 'model_path':  # Keep model path
                del st.session_state[key]
        
        # Reinitialize
        self.initialize_session_state()
        st.success("✅ Session reset successfully!")

def main():
    """Main entry point"""
    app = ForecastingApp()
    app.run()

if __name__ == "__main__":
    main()

