"""
Forecast display component for Streamlit UI
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ForecastDisplayComponent:
    """
    Component for displaying forecast results and comparisons
    """
    
    def __init__(self):
        """Initialize forecast display component"""
        pass
    
    def render_comprehensive_results(self, all_results: Dict[str, Any]):
        """
        Render comprehensive results from all agents
        
        Args:
            all_results: Dictionary with results from all agents
        """
        st.header("📊 Comprehensive Forecast Results")
        
        # Extract results
        agent1_result = all_results.get('agent1_result')
        agent2_result = all_results.get('agent2_result')
        agent3_result = all_results.get('agent3_result')
        forecast_data = all_results.get('forecast_data', {})
        
        # Summary metrics
        self._render_summary_metrics(agent1_result, agent2_result)
        
        # Main forecast comparison chart
        self._render_main_forecast_chart(agent1_result, agent2_result, forecast_data)
        
        # Detailed comparison table
        self._render_comparison_table(agent1_result, agent2_result)
        
        # Individual agent results
        if agent1_result:
            self._render_agent1_details(agent1_result)
        
        if agent2_result:
            self._render_agent2_details(agent2_result)
        
        if agent3_result:
            self._render_agent3_details(agent3_result)
        
        # Performance metrics
        self._render_performance_metrics(all_results)
    
    def _render_summary_metrics(self, agent1_result: Dict[str, Any], agent2_result: Dict[str, Any]):
        """Render summary metrics"""
        st.subheader("📈 Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if agent1_result:
            with col1:
                mape = agent1_result.get('accuracy_metrics', {}).get('mape', 0)
                st.metric("Agent 1 MAPE", f"{mape:.2f}%")
            
            with col2:
                method = agent1_result.get('method_used', 'Unknown')
                st.metric("Best Method", method)
            
            with col3:
                driver_sig = agent1_result.get('driver_significance', {}).get('is_significant', False)
                st.metric("Driver Significant", "Yes" if driver_sig else "No")
        
        if agent2_result:
            with col4:
                periods_changed = agent2_result.get('impact_metrics', {}).get('periods_changed', 0)
                st.metric("Periods Tweaked", periods_changed)
    
    def _render_main_forecast_chart(self, 
                                   agent1_result: Dict[str, Any], 
                                   agent2_result: Dict[str, Any],
                                   forecast_data: Dict[str, Any]):
        """Render main forecast comparison chart"""
        st.subheader("📊 Forecast Comparison Chart")
        
        # Prepare data
        chart_data = {}
        
        # Historical actuals
        if forecast_data.get('actuals'):
            actuals = forecast_data['actuals']
            historical_periods = list(range(-len(actuals), 0))
            chart_data['Historical Actuals'] = {
                'x': historical_periods,
                'y': actuals,
                'type': 'historical'
            }
        
        # Agent 1 forecast
        if agent1_result and agent1_result.get('forecast_values'):
            forecast_values = agent1_result['forecast_values']
            forecast_periods = list(range(1, len(forecast_values) + 1))
            chart_data['Agent 1 Forecast'] = {
                'x': forecast_periods,
                'y': forecast_values,
                'type': 'forecast'
            }
        
        # Agent 2 forecast
        if agent2_result and agent2_result.get('tweaked_forecast'):
            tweaked_values = agent2_result['tweaked_forecast']
            forecast_periods = list(range(1, len(tweaked_values) + 1))
            chart_data['Agent 2 Forecast'] = {
                'x': forecast_periods,
                'y': tweaked_values,
                'type': 'forecast'
            }
        
        # Create Plotly chart
        fig = go.Figure()
        
        # Color scheme
        colors = {
            'Historical Actuals': '#1f77b4',
            'Agent 1 Forecast': '#ff7f0e',
            'Agent 2 Forecast': '#2ca02c'
        }
        
        for series_name, data in chart_data.items():
            line_style = dict(width=3) if data['type'] == 'historical' else dict(width=2, dash='dash')
            
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                name=series_name,
                line=dict(color=colors.get(series_name, '#d62728'), **line_style),
                marker=dict(size=6)
            ))
        
        # Add vertical line at forecast start
        if chart_data:
            fig.add_vline(x=0, line_dash="dot", line_color="gray", 
                         annotation_text="Forecast Start", annotation_position="top")
        
        # Update layout
        fig.update_layout(
            title="Historical Data vs Forecasts",
            xaxis_title="Time Period",
            yaxis_title="ACD Call Volume",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_table(self, agent1_result: Dict[str, Any], agent2_result: Dict[str, Any]):
        """Render detailed comparison table"""
        st.subheader("📋 Detailed Forecast Comparison")
        
        if not agent1_result or not agent1_result.get('forecast_values'):
            st.warning("No forecast data available for comparison")
            return
        
        # Prepare comparison data
        agent1_values = agent1_result['forecast_values']
        periods = list(range(1, len(agent1_values) + 1))
        
        comparison_data = {
            'Period': periods,
            'Agent 1 (Original)': agent1_values
        }
        
        if agent2_result and agent2_result.get('tweaked_forecast'):
            agent2_values = agent2_result['tweaked_forecast']
            comparison_data['Agent 2 (Tweaked)'] = agent2_values
            
            # Calculate differences
            differences = [a2 - a1 for a1, a2 in zip(agent1_values, agent2_values)]
            percentage_changes = [(diff / a1 * 100) if a1 != 0 else 0 for diff, a1 in zip(differences, agent1_values)]
            
            comparison_data['Difference'] = differences
            comparison_data['% Change'] = [f"{pc:+.1f}%" for pc in percentage_changes]
        
        # Create DataFrame and display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        styled_df = comparison_df.style.format({
            'Agent 1 (Original)': '{:.1f}',
            'Agent 2 (Tweaked)': '{:.1f}',
            'Difference': '{:+.1f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics
        if 'Difference' in comparison_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_diff = sum(comparison_data['Difference']) / len(comparison_data['Difference'])
                st.metric("Avg Difference", f"{avg_diff:+.1f}")
            
            with col2:
                max_diff = max(comparison_data['Difference'], key=abs)
                st.metric("Max Difference", f"{max_diff:+.1f}")
            
            with col3:
                total_original = sum(comparison_data['Agent 1 (Original)'])
                total_tweaked = sum(comparison_data['Agent 2 (Tweaked)'])
                net_change = ((total_tweaked - total_original) / total_original * 100) if total_original != 0 else 0
                st.metric("Net Change", f"{net_change:+.1f}%")
            
            with col4:
                periods_changed = sum(1 for diff in comparison_data['Difference'] if abs(diff) > 0.1)
                st.metric("Periods Changed", periods_changed)
    
    def _render_agent1_details(self, agent1_result: Dict[str, Any]):
        """Render Agent 1 detailed results"""
        with st.expander("🎯 Agent 1 - Detailed Results", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                accuracy = agent1_result.get('accuracy_metrics', {})
                
                metrics_data = {
                    'Metric': ['MAPE', 'Correlation', 'Bias', 'RMSE'],
                    'Value': [
                        f"{accuracy.get('mape', 0):.2f}%",
                        f"{accuracy.get('correlation', 0):.3f}",
                        f"{accuracy.get('bias', 0):.1f}",
                        f"{accuracy.get('rmse', 0):.1f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
            
            with col2:
                st.subheader("Driver Analysis")
                driver_sig = agent1_result.get('driver_significance', {})
                
                driver_data = {
                    'Aspect': ['Significant', 'Correlation', 'P-value'],
                    'Value': [
                        'Yes' if driver_sig.get('is_significant', False) else 'No',
                        f"{driver_sig.get('correlation', 0):.3f}",
                        f"{driver_sig.get('p_value', 1):.4f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(driver_data), hide_index=True)
            
            # Model selection details
            if 'all_models_tested' in agent1_result:
                st.subheader("Model Comparison")
                models_tested = agent1_result['all_models_tested']
                
                model_comparison = []
                for model_name, model_result in models_tested.items():
                    model_comparison.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'MAPE': f"{model_result.get('accuracy', {}).get('mape', 0):.2f}%",
                        'Selected': '✅' if model_name == agent1_result.get('method_name', '') else ''
                    })
                
                st.dataframe(pd.DataFrame(model_comparison), hide_index=True)
    
    def _render_agent2_details(self, agent2_result: Dict[str, Any]):
        """Render Agent 2 detailed results"""
        with st.expander("🔧 Agent 2 - Tweaking Details", expanded=False):
            # Human instruction
            st.subheader("Human Instruction")
            instruction = agent2_result.get('human_instruction', 'No instruction provided')
            st.text_area("Instruction", instruction, height=100, disabled=True)
            
            # Impact analysis
            st.subheader("Impact Analysis")
            impact = agent2_result.get('impact_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Periods Changed", impact.get('periods_changed', 0))
            with col2:
                st.metric("Avg % Change", f"{impact.get('average_percentage_change', 0):.2f}%")
            with col3:
                st.metric("Max Change", f"{impact.get('max_percentage_change', 0):.2f}%")
            
            # Adjustments made
            if 'tweaking_details' in agent2_result:
                adjustments = agent2_result['tweaking_details'].get('adjustments_made', [])
                
                if adjustments and not any('error' in adj for adj in adjustments):
                    st.subheader("Adjustments Made")
                    
                    adj_data = []
                    for adj in adjustments:
                        adj_data.append({
                            'Period': adj.get('period_index', 0) + 1,
                            'Original': f"{adj.get('original_value', 0):.1f}",
                            'Adjustment': f"{adj.get('adjustment', 0):+.1f}",
                            'New Value': f"{adj.get('new_value', 0):.1f}",
                            '% Change': f"{adj.get('percentage_change', 0):+.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(adj_data), hide_index=True)
    
    def _render_agent3_details(self, agent3_result: Dict[str, Any]):
        """Render Agent 3 detailed results"""
        with st.expander("💡 Agent 3 - Explanations", expanded=False):
            explanation = agent3_result.get('explanation', 'No explanation available')
            
            st.subheader("AI Explanation")
            st.markdown(explanation)
            
            # Key insights
            if 'key_insights' in agent3_result:
                insights = agent3_result['key_insights']
                if insights:
                    st.subheader("Key Insights")
                    for insight in insights:
                        st.write(f"• {insight}")
    
    def _render_performance_metrics(self, all_results: Dict[str, Any]):
        """Render system performance metrics"""
        with st.expander("⚡ System Performance", expanded=False):
            workflow_progress = all_results.get('workflow_progress', {})
            session_summary = all_results.get('session_summary', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Workflow Status")
                st.write(f"**Current Step:** {workflow_progress.get('current_step', 'Unknown')}")
                st.write(f"**Progress:** {workflow_progress.get('progress_percentage', 0):.0f}%")
                st.write(f"**Completed Steps:** {len(workflow_progress.get('completed_steps', []))}")
            
            with col2:
                st.subheader("Session Info")
                st.write(f"**Session ID:** {session_summary.get('session_id', 'Unknown')}")
                st.write(f"**Active Agents:** {len(session_summary.get('active_agents', []))}")
                st.write(f"**History Entries:** {session_summary.get('history_entries', 0)}")
    
    def render_forecast_chart(self, 
                            forecast_values: List[float], 
                            title: str = "Forecast",
                            historical_data: Optional[List[float]] = None) -> go.Figure:
        """
        Render a single forecast chart
        
        Args:
            forecast_values: Forecast values to plot
            title: Chart title
            historical_data: Optional historical data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical data if provided
        if historical_data:
            historical_periods = list(range(-len(historical_data), 0))
            fig.add_trace(go.Scatter(
                x=historical_periods,
                y=historical_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6)
            ))
        
        # Add forecast data
        forecast_periods = list(range(1, len(forecast_values) + 1))
        fig.add_trace(go.Scatter(
            x=forecast_periods,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add vertical line at forecast start
        if historical_data:
            fig.add_vline(x=0, line_dash="dot", line_color="gray", 
                         annotation_text="Forecast Start", annotation_position="top")
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="ACD Call Volume",
            hovermode='x unified',
            height=400
        )
        
        return fig

