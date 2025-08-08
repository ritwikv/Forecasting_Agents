"""
Chart generation utilities for the Streamlit UI
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Utility class for generating various charts and visualizations
    """
    
    def __init__(self):
        """Initialize chart generator"""
        self.color_palette = {
            'historical': '#1f77b4',
            'agent1': '#ff7f0e', 
            'agent2': '#2ca02c',
            'driver': '#d62728',
            'fitted': '#9467bd',
            'confidence': 'rgba(255, 127, 14, 0.2)'
        }
    
    def create_forecast_comparison_chart(self, 
                                       historical_data: Optional[List[float]] = None,
                                       agent1_forecast: Optional[List[float]] = None,
                                       agent2_forecast: Optional[List[float]] = None,
                                       driver_forecast: Optional[List[float]] = None,
                                       title: str = "Forecast Comparison") -> go.Figure:
        """
        Create comprehensive forecast comparison chart
        
        Args:
            historical_data: Historical actual values
            agent1_forecast: Agent 1 forecast values
            agent2_forecast: Agent 2 forecast values  
            driver_forecast: Driver forecast values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical data
        if historical_data:
            historical_periods = list(range(-len(historical_data), 0))
            fig.add_trace(go.Scatter(
                x=historical_periods,
                y=historical_data,
                mode='lines+markers',
                name='Historical Actuals',
                line=dict(color=self.color_palette['historical'], width=3),
                marker=dict(size=6),
                hovertemplate='<b>Historical</b><br>Period: %{x}<br>Value: %{y:.1f}<extra></extra>'
            ))
        
        # Add Agent 1 forecast
        if agent1_forecast:
            forecast_periods = list(range(1, len(agent1_forecast) + 1))
            fig.add_trace(go.Scatter(
                x=forecast_periods,
                y=agent1_forecast,
                mode='lines+markers',
                name='Agent 1 (Statistical)',
                line=dict(color=self.color_palette['agent1'], width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='<b>Agent 1</b><br>Period: %{x}<br>Value: %{y:.1f}<extra></extra>'
            ))
        
        # Add Agent 2 forecast
        if agent2_forecast:
            forecast_periods = list(range(1, len(agent2_forecast) + 1))
            fig.add_trace(go.Scatter(
                x=forecast_periods,
                y=agent2_forecast,
                mode='lines+markers',
                name='Agent 2 (Tweaked)',
                line=dict(color=self.color_palette['agent2'], width=2, dash='dot'),
                marker=dict(size=6),
                hovertemplate='<b>Agent 2</b><br>Period: %{x}<br>Value: %{y:.1f}<extra></extra>'
            ))
        
        # Add driver forecast for comparison
        if driver_forecast and historical_data:
            # Show driver forecast for historical periods
            historical_periods = list(range(-len(driver_forecast), 0))
            fig.add_trace(go.Scatter(
                x=historical_periods,
                y=driver_forecast,
                mode='lines',
                name='Driver Forecast',
                line=dict(color=self.color_palette['driver'], width=1, dash='dashdot'),
                opacity=0.7,
                hovertemplate='<b>Driver</b><br>Period: %{x}<br>Value: %{y:.1f}<extra></extra>'
            ))
        
        # Add vertical line at forecast start
        if historical_data and (agent1_forecast or agent2_forecast):
            fig.add_vline(
                x=0, 
                line_dash="dot", 
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top"
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Time Period",
            yaxis_title="ACD Call Volume",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_accuracy_comparison_chart(self, 
                                       agent1_metrics: Dict[str, float],
                                       agent2_metrics: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Create accuracy metrics comparison chart
        
        Args:
            agent1_metrics: Agent 1 accuracy metrics
            agent2_metrics: Agent 2 accuracy metrics (optional)
            
        Returns:
            Plotly figure
        """
        metrics = ['MAPE', 'Correlation', 'Bias %']
        
        agent1_values = [
            agent1_metrics.get('mape', 0),
            agent1_metrics.get('correlation', 0) * 100,  # Convert to percentage
            agent1_metrics.get('bias_percentage', 0)
        ]
        
        fig = go.Figure()
        
        # Agent 1 bars
        fig.add_trace(go.Bar(
            name='Agent 1',
            x=metrics,
            y=agent1_values,
            marker_color=self.color_palette['agent1'],
            text=[f'{v:.1f}' for v in agent1_values],
            textposition='auto'
        ))
        
        # Agent 2 bars (if available)
        if agent2_metrics:
            agent2_values = [
                agent2_metrics.get('mape', 0),
                agent2_metrics.get('correlation', 0) * 100,
                agent2_metrics.get('bias_percentage', 0)
            ]
            
            fig.add_trace(go.Bar(
                name='Agent 2',
                x=metrics,
                y=agent2_values,
                marker_color=self.color_palette['agent2'],
                text=[f'{v:.1f}' for v in agent2_values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Forecast Accuracy Comparison",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_residuals_chart(self, 
                             actual: List[float], 
                             fitted: List[float],
                             title: str = "Residuals Analysis") -> go.Figure:
        """
        Create residuals analysis chart
        
        Args:
            actual: Actual values
            fitted: Fitted values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        residuals = [a - f for a, f in zip(actual, fitted)]
        periods = list(range(1, len(residuals) + 1))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Time', 'Residuals vs Fitted', 
                          'Residuals Distribution', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Time
        fig.add_trace(
            go.Scatter(x=periods, y=residuals, mode='markers+lines',
                      name='Residuals', marker=dict(color=self.color_palette['agent1'])),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(x=fitted, y=residuals, mode='markers',
                      name='Residuals vs Fitted', marker=dict(color=self.color_palette['agent2'])),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # Residuals Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribution', 
                        marker=dict(color=self.color_palette['historical'])),
            row=2, col=1
        )
        
        # Q-Q Plot (simplified)
        sorted_residuals = sorted(residuals)
        theoretical_quantiles = np.linspace(-2, 2, len(sorted_residuals))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers',
                      name='Q-Q Plot', marker=dict(color=self.color_palette['driver'])),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text=title, showlegend=False)
        
        return fig
    
    def create_forecast_uncertainty_chart(self, 
                                        forecast: List[float],
                                        confidence_intervals: Optional[List[Tuple[float, float]]] = None,
                                        title: str = "Forecast with Uncertainty") -> go.Figure:
        """
        Create forecast chart with uncertainty bands
        
        Args:
            forecast: Forecast values
            confidence_intervals: List of (lower, upper) confidence intervals
            title: Chart title
            
        Returns:
            Plotly figure
        """
        periods = list(range(1, len(forecast) + 1))
        
        fig = go.Figure()
        
        # Add confidence intervals if provided
        if confidence_intervals:
            lower_bounds = [ci[0] for ci in confidence_intervals]
            upper_bounds = [ci[1] for ci in confidence_intervals]
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=periods + periods[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor=self.color_palette['confidence'],
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                hoverinfo="skip"
            ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=periods,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color=self.color_palette['agent1'], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="ACD Call Volume",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_model_comparison_chart(self, models_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create model comparison chart showing MAPE for different models
        
        Args:
            models_results: Dictionary of model results
            
        Returns:
            Plotly figure
        """
        model_names = []
        mape_values = []
        colors = []
        
        color_map = {
            'exponential_smoothing': '#1f77b4',
            'linear_trend': '#ff7f0e',
            'seasonal_naive': '#2ca02c',
            'ensemble': '#d62728',
            'driver_enhanced': '#9467bd'
        }
        
        for model_name, result in models_results.items():
            model_names.append(model_name.replace('_', ' ').title())
            mape = result.get('accuracy', {}).get('mape', 0)
            mape_values.append(mape)
            colors.append(color_map.get(model_name, '#7f7f7f'))
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=mape_values,
                marker_color=colors,
                text=[f'{v:.2f}%' for v in mape_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Model Performance Comparison (MAPE)",
            xaxis_title="Models",
            yaxis_title="MAPE (%)",
            height=400
        )
        
        return fig
    
    def create_seasonal_decomposition_chart(self, 
                                          data: List[float],
                                          period: int = 12) -> go.Figure:
        """
        Create seasonal decomposition chart
        
        Args:
            data: Time series data
            period: Seasonal period
            
        Returns:
            Plotly figure
        """
        # Simple seasonal decomposition
        n = len(data)
        periods = list(range(1, n + 1))
        
        # Calculate trend (moving average)
        trend = []
        for i in range(n):
            start = max(0, i - period // 2)
            end = min(n, i + period // 2 + 1)
            trend.append(sum(data[start:end]) / (end - start))
        
        # Calculate seasonal component
        seasonal = []
        for i in range(n):
            seasonal_index = i % period
            seasonal_values = [data[j] - trend[j] for j in range(seasonal_index, n, period) if j < len(trend)]
            seasonal.append(sum(seasonal_values) / len(seasonal_values) if seasonal_values else 0)
        
        # Calculate residual
        residual = [data[i] - trend[i] - seasonal[i] for i in range(n)]
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Original data
        fig.add_trace(go.Scatter(x=periods, y=data, mode='lines', name='Original'), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(x=periods, y=trend, mode='lines', name='Trend'), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(x=periods, y=seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(x=periods, y=residual, mode='lines', name='Residual'), row=4, col=1)
        
        fig.update_layout(height=800, title_text="Seasonal Decomposition", showlegend=False)
        
        return fig
    
    def create_impact_analysis_chart(self, 
                                   original_forecast: List[float],
                                   tweaked_forecast: List[float]) -> go.Figure:
        """
        Create impact analysis chart showing the effect of tweaks
        
        Args:
            original_forecast: Original forecast values
            tweaked_forecast: Tweaked forecast values
            
        Returns:
            Plotly figure
        """
        periods = list(range(1, len(original_forecast) + 1))
        differences = [t - o for o, t in zip(original_forecast, tweaked_forecast)]
        percentage_changes = [(d / o * 100) if o != 0 else 0 for d, o in zip(differences, original_forecast)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Absolute Impact', 'Percentage Impact'),
            vertical_spacing=0.15
        )
        
        # Absolute differences
        colors = ['red' if d < 0 else 'green' for d in differences]
        fig.add_trace(
            go.Bar(x=periods, y=differences, marker_color=colors, name='Absolute Change'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Percentage changes
        colors = ['red' if p < 0 else 'green' for p in percentage_changes]
        fig.add_trace(
            go.Bar(x=periods, y=percentage_changes, marker_color=colors, name='Percentage Change'),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=500,
            title_text="Tweaking Impact Analysis",
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Absolute Change", row=1, col=1)
        fig.update_yaxes(title_text="Percentage Change (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time Period", row=2, col=1)
        
        return fig

