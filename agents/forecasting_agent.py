"""
Agent 1: Time Series Forecasting Agent
Minimizes MAPE and uses Driver_Forecast when statistically significant
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np

from .base_agent import BaseAgent
from utils.forecasting_utils import ForecastingUtils
from utils.statistical_tests import StatisticalTests
from config import AGENT_CONFIG, DATA_CONFIG

logger = logging.getLogger(__name__)

class ForecastingAgent(BaseAgent):
    """
    Agent 1: Time Series Forecasting Agent
    
    Responsibilities:
    - Forecast ACD_Call_Volume_Forecast using ACD Call Volume Actuals
    - Use Driver_Forecast only if statistically significant
    - Minimize Mean Absolute Percentage Error (MAPE)
    - Recommend forecast horizon based on data availability
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Forecasting Agent
        
        Args:
            model_path: Path to the model file
        """
        super().__init__(
            agent_name=AGENT_CONFIG["forecasting_agent"]["name"],
            model_path=model_path
        )
        
        # Initialize utilities
        self.forecasting_utils = ForecastingUtils()
        self.statistical_tests = StatisticalTests(
            significance_level=AGENT_CONFIG["forecasting_agent"]["significance_threshold"]
        )
        
        # Agent-specific configuration
        self.config = AGENT_CONFIG["forecasting_agent"]
        self.target_mape = self.config["mape_target"]
        self.max_forecast_months = self.config["max_forecast_months"]
        self.min_data_points = self.config["min_data_points"]
        
        # State variables
        self.last_forecast_result = None
        self.driver_significance_result = None
        self.best_model_info = None
    
    def _get_default_system_message(self) -> str:
        """Get default system message for the forecasting agent"""
        return f"""You are Agent 1, a specialized Time Series Forecasting Agent for ACD Call Volume prediction.

Your primary responsibilities:
1. Forecast ACD_Call_Volume_Forecast using historical ACD Call Volume Actuals
2. Use Driver_Forecast ONLY if it shows statistical significance (p < {self.config['significance_threshold']})
3. Minimize Mean Absolute Percentage Error (MAPE) - target is {self.target_mape}%
4. Recommend appropriate forecast horizon based on available data
5. Provide clear explanations of your forecasting methodology

Key principles:
- Always test Driver_Forecast significance before using it
- Use ensemble methods when appropriate to minimize MAPE
- Be conservative with forecast horizons for limited data
- Explain your model selection rationale
- Provide confidence intervals when possible

When asked about forecast periods, analyze the data quality and recommend the maximum reliable forecast horizon.
Always explain your reasoning in terms of statistical significance, data sufficiency, and expected accuracy."""
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process forecasting request
        
        Args:
            input_data: Dictionary containing:
                - actuals: List of ACD Call Volume Actuals
                - driver_forecast: List of Driver_Forecast values
                - forecast_periods: Number of periods to forecast (optional)
                - target_year: Target year for forecast (optional)
                - target_month: Target month for forecast (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with forecasting results
        """
        try:
            logger.info(f"Processing forecasting request for {self.agent_name}")
            
            # Extract input data
            actuals = input_data.get('actuals', [])
            driver_forecast = input_data.get('driver_forecast', [])
            forecast_periods = input_data.get('forecast_periods')
            target_year = input_data.get('target_year')
            target_month = input_data.get('target_month')
            
            # Validate input data
            validation_result = self._validate_input_data(actuals, driver_forecast)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'agent': self.agent_name
                }
            
            # Determine forecast periods
            if forecast_periods is None:
                forecast_periods = self._determine_forecast_periods(
                    len(actuals), target_year, target_month
                )
            
            # Test driver forecast significance
            driver_significance = self.statistical_tests.test_driver_significance(
                actuals, driver_forecast
            )
            self.driver_significance_result = driver_significance
            
            # Generate forecasts using different methods
            forecast_results = self._generate_forecasts(
                actuals, driver_forecast, forecast_periods, driver_significance
            )
            
            # Select best model based on MAPE
            best_model = self._select_best_model(forecast_results)
            self.best_model_info = best_model
            
            # Generate AI explanation
            ai_explanation = self._generate_ai_explanation(
                actuals, driver_forecast, forecast_periods, 
                driver_significance, best_model
            )
            
            # Prepare final result
            result = {
                'success': True,
                'agent': self.agent_name,
                'forecast_periods': forecast_periods,
                'forecast_values': best_model['forecast'],
                'fitted_values': best_model['fitted_values'],
                'method_used': best_model['method'],
                'accuracy_metrics': best_model['accuracy'],
                'driver_significance': driver_significance,
                'model_details': best_model,
                'all_models_tested': forecast_results,
                'ai_explanation': ai_explanation,
                'recommendations': self._generate_recommendations(
                    len(actuals), best_model['accuracy']['mape']
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            self.last_forecast_result = result
            self.last_result = result
            
            logger.info(f"Forecasting completed successfully. Best MAPE: {best_model['accuracy']['mape']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in forecasting process: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_input_data(self, actuals: List[float], driver_forecast: List[float]) -> Dict[str, Any]:
        """Validate input data for forecasting"""
        if not actuals or len(actuals) < self.min_data_points:
            return {
                'is_valid': False,
                'error': f'Insufficient actual data points. Need at least {self.min_data_points}, got {len(actuals)}'
            }
        
        if len(driver_forecast) != len(actuals):
            return {
                'is_valid': False,
                'error': f'Actuals ({len(actuals)}) and driver forecast ({len(driver_forecast)}) lengths do not match'
            }
        
        # Check for valid numeric values
        valid_actuals = [x for x in actuals if x is not None and not np.isnan(x) and np.isfinite(x)]
        if len(valid_actuals) < len(actuals) * 0.8:
            return {
                'is_valid': False,
                'error': f'Too many invalid actual values. Valid: {len(valid_actuals)}/{len(actuals)}'
            }
        
        return {'is_valid': True}
    
    def _determine_forecast_periods(self, data_length: int, target_year: Optional[int], target_month: Optional[int]) -> int:
        """Determine appropriate number of forecast periods"""
        if target_year and target_month:
            # Calculate periods from current date to target
            current_date = datetime.now()
            target_date = datetime(target_year, target_month, 1)
            
            if target_date <= current_date:
                return 1  # Default to 1 period if target is in the past
            
            months_diff = (target_date.year - current_date.year) * 12 + (target_date.month - current_date.month)
            return min(months_diff, self.max_forecast_months)
        
        # Use recommendation based on data length
        recommendations = self.forecasting_utils.recommend_forecast_periods(data_length)
        return min(recommendations['max_recommended_periods'], self.max_forecast_months)
    
    def _generate_forecasts(self, 
                          actuals: List[float], 
                          driver_forecast: List[float],
                          forecast_periods: int,
                          driver_significance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts using multiple methods"""
        forecast_results = {}
        
        # Method 1: Simple Exponential Smoothing
        try:
            result = self.forecasting_utils.simple_exponential_smoothing(
                actuals, forecast_periods=forecast_periods
            )
            forecast_results['exponential_smoothing'] = result
        except Exception as e:
            logger.warning(f"Exponential smoothing failed: {str(e)}")
        
        # Method 2: Linear Trend
        try:
            result = self.forecasting_utils.linear_trend_forecast(
                actuals, forecast_periods=forecast_periods
            )
            forecast_results['linear_trend'] = result
        except Exception as e:
            logger.warning(f"Linear trend failed: {str(e)}")
        
        # Method 3: Seasonal Naive
        try:
            result = self.forecasting_utils.seasonal_naive_forecast(
                actuals, forecast_periods=forecast_periods
            )
            forecast_results['seasonal_naive'] = result
        except Exception as e:
            logger.warning(f"Seasonal naive failed: {str(e)}")
        
        # Method 4: Ensemble
        try:
            result = self.forecasting_utils.ensemble_forecast(
                actuals, forecast_periods=forecast_periods
            )
            forecast_results['ensemble'] = result
        except Exception as e:
            logger.warning(f"Ensemble forecast failed: {str(e)}")
        
        # Method 5: Driver-Enhanced (if significant)
        if driver_significance['is_significant']:
            try:
                result = self._driver_enhanced_forecast(
                    actuals, driver_forecast, forecast_periods
                )
                forecast_results['driver_enhanced'] = result
            except Exception as e:
                logger.warning(f"Driver-enhanced forecast failed: {str(e)}")
        
        return forecast_results
    
    def _driver_enhanced_forecast(self, 
                                actuals: List[float],
                                driver_forecast: List[float],
                                forecast_periods: int) -> Dict[str, Any]:
        """Generate forecast using driver forecast as additional input"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Use driver forecast as predictor
        X = np.array(driver_forecast).reshape(-1, 1)
        y = np.array(actuals)
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate fitted values
        fitted_values = model.predict(X)
        
        # For forecast, we need future driver values
        # Use the last known driver value or trend
        if len(driver_forecast) > 0:
            last_driver = driver_forecast[-1]
            # Simple assumption: driver forecast remains constant or follows trend
            if len(driver_forecast) > 1:
                driver_trend = driver_forecast[-1] - driver_forecast[-2]
                future_drivers = [last_driver + driver_trend * (i + 1) for i in range(forecast_periods)]
            else:
                future_drivers = [last_driver] * forecast_periods
            
            # Generate forecast
            future_X = np.array(future_drivers).reshape(-1, 1)
            forecast = model.predict(future_X)
        else:
            forecast = [np.mean(actuals)] * forecast_periods
        
        # Calculate accuracy
        accuracy = self.forecasting_utils.calculate_forecast_accuracy_metrics(
            actuals, fitted_values.tolist()
        )
        
        return {
            'method': 'Driver Enhanced Regression',
            'parameters': {
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(r2_score(y, fitted_values))
            },
            'fitted_values': fitted_values.tolist(),
            'forecast': forecast.tolist(),
            'accuracy': accuracy,
            'future_drivers_used': future_drivers
        }
    
    def _select_best_model(self, forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select best model based on MAPE"""
        if not forecast_results:
            raise ValueError("No forecast results available for model selection")
        
        best_model = None
        best_mape = float('inf')
        
        for method_name, result in forecast_results.items():
            mape = result['accuracy']['mape']
            if mape < best_mape:
                best_mape = mape
                best_model = result.copy()
                best_model['method_name'] = method_name
        
        if best_model is None:
            # Fallback to first available model
            first_method = list(forecast_results.keys())[0]
            best_model = forecast_results[first_method].copy()
            best_model['method_name'] = first_method
        
        return best_model
    
    def _generate_ai_explanation(self, 
                               actuals: List[float],
                               driver_forecast: List[float],
                               forecast_periods: int,
                               driver_significance: Dict[str, Any],
                               best_model: Dict[str, Any]) -> str:
        """Generate AI explanation of the forecasting process"""
        
        # Prepare context for AI explanation
        context = f"""
        Forecasting Analysis Summary:
        - Historical data points: {len(actuals)}
        - Forecast periods requested: {forecast_periods}
        - Best model selected: {best_model.get('method', 'Unknown')}
        - Achieved MAPE: {best_model['accuracy']['mape']:.2f}%
        - Target MAPE: {self.target_mape}%
        
        Driver Forecast Significance Test:
        - Is significant: {driver_significance['is_significant']}
        - Correlation: {driver_significance.get('correlation', 0):.3f}
        - P-value: {driver_significance.get('p_value', 1):.4f}
        
        Model Performance:
        - Mean Actual: {best_model['accuracy']['mean_actual']:.1f}
        - Mean Forecast: {best_model['accuracy']['mean_forecast']:.1f}
        - Bias: {best_model['accuracy']['bias']:.1f}
        - Correlation: {best_model['accuracy']['correlation']:.3f}
        """
        
        prompt = f"""Based on the following forecasting analysis, provide a clear explanation of:
        1. Why this forecasting method was selected
        2. How the Driver_Forecast significance test influenced the decision
        3. The expected accuracy and reliability of the forecast
        4. Any limitations or considerations for the forecast
        
        {context}
        
        Provide a professional, technical explanation suitable for business stakeholders."""
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                return response['text']
            else:
                return self._generate_fallback_explanation(best_model, driver_significance)
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {str(e)}")
            return self._generate_fallback_explanation(best_model, driver_significance)
    
    def _generate_fallback_explanation(self, best_model: Dict[str, Any], driver_significance: Dict[str, Any]) -> str:
        """Generate fallback explanation without AI model"""
        explanation = f"""
        Forecasting Method Selected: {best_model.get('method', 'Unknown')}
        
        Model Selection Rationale:
        This method was selected because it achieved the lowest Mean Absolute Percentage Error (MAPE) of {best_model['accuracy']['mape']:.2f}% among all tested methods.
        
        Driver Forecast Analysis:
        The Driver_Forecast variable was {'included' if driver_significance['is_significant'] else 'excluded'} from the final model.
        - Statistical significance: {'Yes' if driver_significance['is_significant'] else 'No'}
        - Correlation with actuals: {driver_significance.get('correlation', 0):.3f}
        - P-value: {driver_significance.get('p_value', 1):.4f}
        
        Forecast Reliability:
        {'The forecast is considered reliable' if best_model['accuracy']['mape'] < self.target_mape else 'The forecast has higher uncertainty than target'} with a MAPE of {best_model['accuracy']['mape']:.2f}% (target: {self.target_mape}%).
        
        Key Considerations:
        - Based on {len(best_model.get('fitted_values', []))} historical data points
        - Forecast correlation with actuals: {best_model['accuracy']['correlation']:.3f}
        - Model bias: {best_model['accuracy']['bias']:.1f}
        """
        
        return explanation.strip()
    
    def _generate_recommendations(self, data_length: int, achieved_mape: float) -> Dict[str, Any]:
        """Generate recommendations based on forecast results"""
        recommendations = {
            'data_quality': [],
            'forecast_horizon': [],
            'model_improvement': [],
            'business_considerations': []
        }
        
        # Data quality recommendations
        if data_length < 24:
            recommendations['data_quality'].append(
                "Consider collecting more historical data for improved forecast accuracy"
            )
        
        # Forecast horizon recommendations
        horizon_rec = self.forecasting_utils.recommend_forecast_periods(data_length)
        recommendations['forecast_horizon'].append(
            f"Recommended maximum forecast horizon: {horizon_rec['max_recommended_periods']} periods"
        )
        recommendations['forecast_horizon'].append(
            f"Confidence level: {horizon_rec['confidence_level']}"
        )
        
        # Model improvement recommendations
        if achieved_mape > self.target_mape:
            recommendations['model_improvement'].append(
                f"MAPE ({achieved_mape:.2f}%) exceeds target ({self.target_mape}%). Consider additional data sources or model refinement."
            )
        
        if not self.driver_significance_result['is_significant']:
            recommendations['model_improvement'].append(
                "Driver_Forecast is not statistically significant. Consider alternative leading indicators."
            )
        
        # Business considerations
        recommendations['business_considerations'].append(
            "Monitor forecast performance and update model as new data becomes available"
        )
        
        if achieved_mape > 20:
            recommendations['business_considerations'].append(
                "High forecast uncertainty - consider scenario planning and regular forecast updates"
            )
        
        return recommendations
    
    def ask_forecast_question(self, question: str) -> Dict[str, Any]:
        """
        Answer questions about the forecast
        
        Args:
            question: Question about the forecast
            
        Returns:
            Dictionary with answer
        """
        if not self.last_forecast_result:
            return {
                'success': False,
                'error': 'No forecast available. Please run a forecast first.',
                'agent': self.agent_name
            }
        
        # Prepare context about the last forecast
        context = f"""
        Last Forecast Results:
        - Method used: {self.last_forecast_result['method_used']}
        - MAPE achieved: {self.last_forecast_result['accuracy_metrics']['mape']:.2f}%
        - Forecast periods: {self.last_forecast_result['forecast_periods']}
        - Driver significance: {self.last_forecast_result['driver_significance']['is_significant']}
        - Forecast values: {self.last_forecast_result['forecast_values']}
        
        Question: {question}
        
        Please provide a detailed answer based on the forecast results and methodology used.
        """
        
        try:
            response = self.chat(context)
            return {
                'success': True,
                'question': question,
                'answer': response['text'],
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error answering forecast question: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': self.agent_name
            }

