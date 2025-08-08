"""
Basic tests for the forecasting agents
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.forecasting_agent import ForecastingAgent
from agents.tweaking_agent import TweakingAgent
from agents.explanation_agent import ExplanationAgent
from utils.forecasting_utils import ForecastingUtils
from utils.statistical_tests import StatisticalTests

class TestForecastingUtils:
    """Test forecasting utilities"""
    
    def setup_method(self):
        """Setup test data"""
        self.forecasting_utils = ForecastingUtils()
        self.sample_data = [100, 110, 105, 120, 115, 130, 125, 140, 135, 150, 145, 160]
    
    def test_calculate_mape(self):
        """Test MAPE calculation"""
        actual = [100, 110, 120]
        forecast = [95, 115, 125]
        
        mape = self.forecasting_utils.calculate_mape(actual, forecast)
        
        assert isinstance(mape, float)
        assert mape > 0
        assert mape < 100  # Should be reasonable percentage
    
    def test_simple_exponential_smoothing(self):
        """Test exponential smoothing"""
        result = self.forecasting_utils.simple_exponential_smoothing(
            self.sample_data, forecast_periods=6
        )
        
        assert result['method'] == 'Simple Exponential Smoothing'
        assert len(result['forecast']) == 6
        assert 'accuracy' in result
        assert 'mape' in result['accuracy']
    
    def test_linear_trend_forecast(self):
        """Test linear trend forecasting"""
        result = self.forecasting_utils.linear_trend_forecast(
            self.sample_data, forecast_periods=6
        )
        
        assert result['method'] == 'Linear Trend'
        assert len(result['forecast']) == 6
        assert 'parameters' in result
        assert 'slope' in result['parameters']
    
    def test_ensemble_forecast(self):
        """Test ensemble forecasting"""
        result = self.forecasting_utils.ensemble_forecast(
            self.sample_data, forecast_periods=6
        )
        
        assert result['method'] == 'Ensemble'
        assert len(result['forecast']) == 6
        assert 'individual_forecasts' in result

class TestStatisticalTests:
    """Test statistical testing utilities"""
    
    def setup_method(self):
        """Setup test data"""
        self.statistical_tests = StatisticalTests()
        np.random.seed(42)
        self.actuals = np.random.normal(100, 10, 50).tolist()
        self.drivers = (np.array(self.actuals) + np.random.normal(0, 5, 50)).tolist()
    
    def test_driver_significance(self):
        """Test driver significance testing"""
        result = self.statistical_tests.test_driver_significance(
            self.actuals, self.drivers
        )
        
        assert 'is_significant' in result
        assert 'p_value' in result
        assert 'correlation' in result
        assert isinstance(result['is_significant'], bool)
        assert 0 <= result['p_value'] <= 1
    
    def test_forecast_residuals(self):
        """Test residual analysis"""
        forecast = [a + np.random.normal(0, 2) for a in self.actuals]
        
        result = self.statistical_tests.test_forecast_residuals(
            self.actuals, forecast
        )
        
        assert 'normality_test' in result
        assert 'autocorrelation_test' in result
        assert 'residual_statistics' in result

class TestForecastingAgent:
    """Test Forecasting Agent"""
    
    def setup_method(self):
        """Setup test agent"""
        # Mock the model to avoid loading actual Phi-3 model
        with patch('agents.forecasting_agent.ChatPhi3Wrapper'):
            self.agent = ForecastingAgent()
            self.agent.is_initialized = True  # Skip actual initialization
    
    def test_validate_input_data(self):
        """Test input data validation"""
        # Valid data
        actuals = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]
        drivers = [95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205]
        
        result = self.agent._validate_input_data(actuals, drivers)
        assert result['is_valid'] is True
        
        # Invalid data - too few points
        short_actuals = [100, 110, 120]
        short_drivers = [95, 105, 115]
        
        result = self.agent._validate_input_data(short_actuals, short_drivers)
        assert result['is_valid'] is False
        assert 'Insufficient' in result['error']
    
    def test_determine_forecast_periods(self):
        """Test forecast period determination"""
        # Test with data length
        periods = self.agent._determine_forecast_periods(24, None, None)
        assert isinstance(periods, int)
        assert periods > 0
        
        # Test with target date
        periods = self.agent._determine_forecast_periods(24, 2025, 12)
        assert isinstance(periods, int)
        assert periods > 0

class TestTweakingAgent:
    """Test Tweaking Agent"""
    
    def setup_method(self):
        """Setup test agent"""
        with patch('agents.tweaking_agent.ChatPhi3Wrapper'):
            self.agent = TweakingAgent()
            self.agent.is_initialized = True
    
    def test_validate_input_data(self):
        """Test input validation"""
        # Valid data
        original_forecast = {'forecast_values': [100, 110, 120]}
        instruction = "Increase December forecast by 10%"
        
        result = self.agent._validate_input_data(original_forecast, instruction)
        assert result['is_valid'] is True
        
        # Invalid data - missing forecast
        result = self.agent._validate_input_data({}, instruction)
        assert result['is_valid'] is False
        
        # Invalid data - empty instruction
        result = self.agent._validate_input_data(original_forecast, "")
        assert result['is_valid'] is False
    
    def test_rule_based_instruction_parsing(self):
        """Test instruction parsing"""
        instruction = "Increase July-25 forecast by 15% due to seasonal factors"
        
        result = self.agent._rule_based_instruction_parsing(instruction)
        
        assert 'target_periods' in result
        assert 'adjustment_direction' in result
        assert 'percentages' in result
        assert result['adjustment_direction'] == 'increase'
        assert 15.0 in result['percentages']

class TestExplanationAgent:
    """Test Explanation Agent"""
    
    def setup_method(self):
        """Setup test agent"""
        with patch('agents.explanation_agent.ChatPhi3Wrapper'):
            self.agent = ExplanationAgent()
            self.agent.is_initialized = True
    
    def test_check_available_results(self):
        """Test result availability checking"""
        # With Agent 1 results
        results = {'agent1': {'forecast_values': [100, 110, 120]}}
        availability = self.agent._check_available_results(results)
        
        assert availability['has_agent1'] is True
        assert availability['has_agent2'] is False
        assert availability['has_any'] is True
        
        # With both agents
        results['agent2'] = {'tweaked_forecast': [105, 115, 125]}
        availability = self.agent._check_available_results(results)
        
        assert availability['has_agent1'] is True
        assert availability['has_agent2'] is True
        assert availability['has_any'] is True

class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow_simulation(self):
        """Test simulated full workflow"""
        # This test simulates the full workflow without actual model loading
        
        # Sample data
        sample_data = {
            'actuals': [100, 110, 105, 120, 115, 130, 125, 140, 135, 150, 145, 160],
            'driver_forecast': [95, 105, 100, 115, 110, 125, 120, 135, 130, 145, 140, 155]
        }
        
        # Test forecasting utilities
        forecasting_utils = ForecastingUtils()
        
        # Test different forecasting methods
        exp_smooth = forecasting_utils.simple_exponential_smoothing(
            sample_data['actuals'], forecast_periods=6
        )
        assert len(exp_smooth['forecast']) == 6
        
        linear_trend = forecasting_utils.linear_trend_forecast(
            sample_data['actuals'], forecast_periods=6
        )
        assert len(linear_trend['forecast']) == 6
        
        ensemble = forecasting_utils.ensemble_forecast(
            sample_data['actuals'], forecast_periods=6
        )
        assert len(ensemble['forecast']) == 6
        
        # Test statistical significance
        stat_tests = StatisticalTests()
        significance = stat_tests.test_driver_significance(
            sample_data['actuals'], sample_data['driver_forecast']
        )
        assert 'is_significant' in significance
        
        print("✅ Integration test passed - all components working together")

if __name__ == "__main__":
    # Run basic tests
    test_integration = TestIntegration()
    test_integration.test_full_workflow_simulation()
    
    test_utils = TestForecastingUtils()
    test_utils.setup_method()
    test_utils.test_calculate_mape()
    test_utils.test_simple_exponential_smoothing()
    
    print("✅ All basic tests passed!")

