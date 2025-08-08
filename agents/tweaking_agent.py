"""
Agent 2: Forecast Tweaking Agent
Processes human instructions to modify forecasts from Agent 1
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
import re

from .base_agent import BaseAgent
from utils.forecasting_utils import ForecastingUtils
from config import AGENT_CONFIG

logger = logging.getLogger(__name__)

class TweakingAgent(BaseAgent):
    """
    Agent 2: Forecast Tweaking Agent
    
    Responsibilities:
    - Process human instructions (up to 300 words) to modify forecasts
    - Understand various tweaking scenarios (comparison with drivers, averages, previous months)
    - Apply intelligent adjustments to Agent 1's forecasts
    - Explain the tweaking rationale and outcomes
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Tweaking Agent
        
        Args:
            model_path: Path to the model file
        """
        super().__init__(
            agent_name=AGENT_CONFIG["tweaking_agent"]["name"],
            model_path=model_path
        )
        
        # Initialize utilities
        self.forecasting_utils = ForecastingUtils()
        
        # Agent-specific configuration
        self.config = AGENT_CONFIG["tweaking_agent"]
        self.max_instruction_length = self.config["max_instruction_length"]
        self.adjustment_limit = self.config["adjustment_limit"]
        
        # State variables
        self.original_forecast = None
        self.tweaked_forecast = None
        self.last_instruction = None
        self.tweak_history = []
    
    def _get_default_system_message(self) -> str:
        """Get default system message for the tweaking agent"""
        return f"""You are Agent 2, a specialized Forecast Tweaking Agent that modifies forecasts based on human business insights.

Your primary responsibilities:
1. Parse human instructions (up to {self.max_instruction_length} words) to understand desired forecast adjustments
2. Apply intelligent tweaks to forecasts from Agent 1
3. Handle various comparison scenarios:
   - Comparisons with Driver_Forecast values
   - Comparisons with historical averages
   - Comparisons with previous months' forecasts or actuals
   - Percentage-based adjustments
4. Explain your tweaking rationale and the resulting changes
5. Ensure adjustments are reasonable and business-logical

Key principles:
- Understand the business context behind human instructions
- Apply adjustments that maintain forecast integrity
- Limit extreme adjustments (max {self.adjustment_limit*100}% change from original)
- Provide clear explanations of what was changed and why
- Consider seasonal patterns and business logic in adjustments

When processing instructions, identify:
- Target time periods (months/years)
- Comparison metrics (drivers, averages, previous values)
- Adjustment direction and magnitude
- Business reasoning behind the request"""
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process tweaking request
        
        Args:
            input_data: Dictionary containing:
                - original_forecast: Original forecast from Agent 1
                - human_instruction: Human instruction for tweaking
                - forecast_data: Additional context data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with tweaking results
        """
        try:
            logger.info(f"Processing tweaking request for {self.agent_name}")
            
            # Extract input data
            original_forecast = input_data.get('original_forecast', {})
            human_instruction = input_data.get('human_instruction', '')
            forecast_data = input_data.get('forecast_data', {})
            
            # Validate input data
            validation_result = self._validate_input_data(original_forecast, human_instruction)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'agent': self.agent_name
                }
            
            # Store original forecast
            self.original_forecast = original_forecast
            self.last_instruction = human_instruction
            
            # Parse human instruction
            instruction_analysis = self._parse_human_instruction(human_instruction, forecast_data)
            
            # Apply tweaks based on instruction
            tweaking_result = self._apply_tweaks(
                original_forecast, instruction_analysis, forecast_data
            )
            
            # Generate explanation
            explanation = self._generate_tweak_explanation(
                original_forecast, tweaking_result, instruction_analysis
            )
            
            # Calculate impact metrics
            impact_metrics = self._calculate_impact_metrics(
                original_forecast['forecast_values'],
                tweaking_result['tweaked_forecast']
            )
            
            # Prepare final result
            result = {
                'success': True,
                'agent': self.agent_name,
                'original_forecast': original_forecast,
                'tweaked_forecast': tweaking_result['tweaked_forecast'],
                'human_instruction': human_instruction,
                'instruction_analysis': instruction_analysis,
                'tweaking_details': tweaking_result,
                'explanation': explanation,
                'impact_metrics': impact_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in history
            self.tweak_history.append(result)
            self.tweaked_forecast = result
            self.last_result = result
            
            logger.info("Forecast tweaking completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in tweaking process: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_input_data(self, original_forecast: Dict, human_instruction: str) -> Dict[str, Any]:
        """Validate input data for tweaking"""
        if not original_forecast or 'forecast_values' not in original_forecast:
            return {
                'is_valid': False,
                'error': 'Original forecast data is missing or invalid'
            }
        
        if not human_instruction or len(human_instruction.strip()) == 0:
            return {
                'is_valid': False,
                'error': 'Human instruction is required for tweaking'
            }
        
        if len(human_instruction.split()) > self.max_instruction_length:
            return {
                'is_valid': False,
                'error': f'Instruction too long. Maximum {self.max_instruction_length} words allowed.'
            }
        
        return {'is_valid': True}
    
    def _parse_human_instruction(self, instruction: str, forecast_data: Dict) -> Dict[str, Any]:
        """Parse human instruction to understand tweaking requirements"""
        
        # Use AI to parse the instruction
        parsing_prompt = f"""
        Parse the following human instruction for forecast tweaking and extract key information:
        
        Instruction: "{instruction}"
        
        Please identify and extract:
        1. Target time period (month/year) mentioned
        2. Comparison type (with drivers, averages, previous months, etc.)
        3. Adjustment direction (higher/lower/increase/decrease)
        4. Adjustment magnitude (percentage or absolute values if mentioned)
        5. Business reasoning behind the request
        
        Provide your analysis in a structured format.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(parsing_prompt)
                ai_analysis = response['text']
            else:
                ai_analysis = "AI parsing not available"
        except Exception as e:
            logger.warning(f"AI instruction parsing failed: {str(e)}")
            ai_analysis = "AI parsing failed"
        
        # Rule-based parsing as backup/supplement
        rule_based_analysis = self._rule_based_instruction_parsing(instruction)
        
        return {
            'original_instruction': instruction,
            'ai_analysis': ai_analysis,
            'rule_based_analysis': rule_based_analysis,
            'parsed_elements': rule_based_analysis  # Use rule-based as primary
        }
    
    def _rule_based_instruction_parsing(self, instruction: str) -> Dict[str, Any]:
        """Rule-based parsing of human instructions"""
        instruction_lower = instruction.lower()
        
        # Extract time periods (months and years)
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december',
                 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        target_periods = []
        for month in months:
            if month in instruction_lower:
                # Look for year after month
                month_pattern = rf'{month}[-\s]*(\d{{2,4}})'
                matches = re.findall(month_pattern, instruction_lower)
                if matches:
                    year = matches[0]
                    if len(year) == 2:
                        year = '20' + year
                    target_periods.append(f"{month.capitalize()}-{year}")
                else:
                    target_periods.append(month.capitalize())
        
        # Extract comparison types
        comparison_types = []
        if any(word in instruction_lower for word in ['driver', 'drivers']):
            comparison_types.append('driver_forecast')
        if any(word in instruction_lower for word in ['average', 'avg', 'mean']):
            comparison_types.append('average')
        if any(word in instruction_lower for word in ['last month', 'previous month', 'prior month']):
            comparison_types.append('previous_month')
        if any(word in instruction_lower for word in ['actual', 'actuals']):
            comparison_types.append('actuals')
        
        # Extract adjustment direction
        adjustment_direction = 'unknown'
        if any(word in instruction_lower for word in ['higher', 'increase', 'up', 'more', 'above']):
            adjustment_direction = 'increase'
        elif any(word in instruction_lower for word in ['lower', 'decrease', 'down', 'less', 'below']):
            adjustment_direction = 'decrease'
        
        # Extract percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, instruction)
        
        # Extract absolute numbers
        number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
        numbers = re.findall(number_pattern, instruction.replace('%', ''))
        
        return {
            'target_periods': target_periods,
            'comparison_types': comparison_types,
            'adjustment_direction': adjustment_direction,
            'percentages': [float(p) for p in percentages],
            'numbers': [float(n.replace(',', '')) for n in numbers],
            'instruction_length': len(instruction.split())
        }
    
    def _apply_tweaks(self, 
                     original_forecast: Dict,
                     instruction_analysis: Dict,
                     forecast_data: Dict) -> Dict[str, Any]:
        """Apply tweaks based on parsed instruction"""
        
        original_values = original_forecast['forecast_values']
        tweaked_values = original_values.copy()
        
        parsed = instruction_analysis['parsed_elements']
        adjustments_made = []
        
        # Get additional data for comparisons
        actuals = forecast_data.get('actuals', [])
        driver_forecast = forecast_data.get('driver_forecast', [])
        
        try:
            # Apply adjustments based on parsed instruction
            if parsed['target_periods'] and parsed['comparison_types']:
                for i, target_period in enumerate(parsed['target_periods']):
                    # For simplicity, assume we're adjusting the first few forecast periods
                    # In a real implementation, you'd map target_period to specific forecast indices
                    
                    if i < len(tweaked_values):
                        original_value = tweaked_values[i]
                        adjustment = self._calculate_adjustment(
                            original_value, parsed, actuals, driver_forecast, i
                        )
                        
                        # Apply adjustment with limits
                        max_change = original_value * self.adjustment_limit
                        adjustment = max(-max_change, min(max_change, adjustment))
                        
                        tweaked_values[i] = max(0, original_value + adjustment)
                        
                        adjustments_made.append({
                            'period_index': i,
                            'target_period': target_period,
                            'original_value': original_value,
                            'adjustment': adjustment,
                            'new_value': tweaked_values[i],
                            'percentage_change': (adjustment / original_value * 100) if original_value != 0 else 0
                        })
            
            # If no specific periods identified, apply general adjustment
            elif parsed['adjustment_direction'] != 'unknown' and parsed['percentages']:
                percentage = parsed['percentages'][0] / 100
                
                for i in range(len(tweaked_values)):
                    original_value = tweaked_values[i]
                    
                    if parsed['adjustment_direction'] == 'increase':
                        adjustment = original_value * percentage
                    else:
                        adjustment = -original_value * percentage
                    
                    # Apply limits
                    max_change = original_value * self.adjustment_limit
                    adjustment = max(-max_change, min(max_change, adjustment))
                    
                    tweaked_values[i] = max(0, original_value + adjustment)
                    
                    adjustments_made.append({
                        'period_index': i,
                        'original_value': original_value,
                        'adjustment': adjustment,
                        'new_value': tweaked_values[i],
                        'percentage_change': (adjustment / original_value * 100) if original_value != 0 else 0
                    })
        
        except Exception as e:
            logger.error(f"Error applying tweaks: {str(e)}")
            # Return original values if tweaking fails
            tweaked_values = original_values.copy()
            adjustments_made = [{'error': str(e)}]
        
        return {
            'tweaked_forecast': tweaked_values,
            'adjustments_made': adjustments_made,
            'total_adjustments': len(adjustments_made)
        }
    
    def _calculate_adjustment(self, 
                            original_value: float,
                            parsed: Dict,
                            actuals: List[float],
                            driver_forecast: List[float],
                            period_index: int) -> float:
        """Calculate specific adjustment based on comparison type"""
        
        adjustment = 0.0
        
        for comparison_type in parsed['comparison_types']:
            if comparison_type == 'driver_forecast' and driver_forecast:
                # Compare with driver forecast
                if period_index < len(driver_forecast):
                    driver_value = driver_forecast[period_index]
                    if parsed['adjustment_direction'] == 'decrease':
                        # Make forecast closer to driver value
                        adjustment = (driver_value - original_value) * 0.5
                    elif parsed['adjustment_direction'] == 'increase':
                        # Make forecast higher than driver
                        if original_value < driver_value:
                            adjustment = (driver_value - original_value) * 1.2
            
            elif comparison_type == 'average' and actuals:
                # Compare with historical average
                avg_actual = np.mean(actuals)
                if parsed['adjustment_direction'] == 'increase':
                    if original_value < avg_actual:
                        adjustment = (avg_actual - original_value) * 0.8
                elif parsed['adjustment_direction'] == 'decrease':
                    if original_value > avg_actual:
                        adjustment = -(original_value - avg_actual) * 0.8
            
            elif comparison_type == 'previous_month' and actuals:
                # Compare with last actual value
                if actuals:
                    last_actual = actuals[-1]
                    if parsed['adjustment_direction'] == 'increase':
                        if original_value < last_actual:
                            adjustment = (last_actual - original_value) * 0.6
                    elif parsed['adjustment_direction'] == 'decrease':
                        if original_value > last_actual:
                            adjustment = -(original_value - last_actual) * 0.6
        
        # Apply percentage adjustments if specified
        if parsed['percentages']:
            percentage = parsed['percentages'][0] / 100
            if parsed['adjustment_direction'] == 'increase':
                adjustment = max(adjustment, original_value * percentage)
            elif parsed['adjustment_direction'] == 'decrease':
                adjustment = min(adjustment, -original_value * percentage)
        
        return adjustment
    
    def _generate_tweak_explanation(self, 
                                  original_forecast: Dict,
                                  tweaking_result: Dict,
                                  instruction_analysis: Dict) -> str:
        """Generate explanation of the tweaking process"""
        
        # Prepare context for AI explanation
        adjustments = tweaking_result['adjustments_made']
        total_adjustments = len([adj for adj in adjustments if 'error' not in adj])
        
        context = f"""
        Forecast Tweaking Analysis:
        - Original instruction: "{instruction_analysis['original_instruction']}"
        - Number of adjustments made: {total_adjustments}
        - Adjustments: {adjustments}
        
        Original forecast method: {original_forecast.get('method_used', 'Unknown')}
        Original MAPE: {original_forecast.get('accuracy_metrics', {}).get('mape', 'Unknown')}
        
        Please explain:
        1. How the human instruction was interpreted
        2. What specific adjustments were made and why
        3. The business logic behind the changes
        4. The expected impact on forecast accuracy
        """
        
        prompt = f"""Based on the following forecast tweaking analysis, provide a clear business explanation of the changes made:
        
        {context}
        
        Provide a professional explanation suitable for business stakeholders, focusing on the rationale and expected outcomes."""
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                return response['text']
            else:
                return self._generate_fallback_explanation(tweaking_result, instruction_analysis)
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {str(e)}")
            return self._generate_fallback_explanation(tweaking_result, instruction_analysis)
    
    def _generate_fallback_explanation(self, tweaking_result: Dict, instruction_analysis: Dict) -> str:
        """Generate fallback explanation without AI model"""
        
        adjustments = tweaking_result['adjustments_made']
        instruction = instruction_analysis['original_instruction']
        
        explanation = f"""
        Forecast Tweaking Summary:
        
        Human Instruction: "{instruction}"
        
        Adjustments Made:
        """
        
        for i, adj in enumerate(adjustments):
            if 'error' not in adj:
                explanation += f"""
        Period {adj.get('period_index', i) + 1}:
        - Original value: {adj.get('original_value', 0):.1f}
        - Adjustment: {adj.get('adjustment', 0):+.1f}
        - New value: {adj.get('new_value', 0):.1f}
        - Change: {adj.get('percentage_change', 0):+.1f}%
        """
        
        explanation += f"""
        
        Rationale:
        The adjustments were made based on the human instruction to align the forecast with business expectations and market insights. The changes maintain forecast integrity while incorporating domain expertise.
        
        Total periods adjusted: {len([adj for adj in adjustments if 'error' not in adj])}
        """
        
        return explanation.strip()
    
    def _calculate_impact_metrics(self, original_values: List[float], tweaked_values: List[float]) -> Dict[str, Any]:
        """Calculate impact metrics of the tweaking"""
        
        if len(original_values) != len(tweaked_values):
            return {'error': 'Mismatched forecast lengths'}
        
        original_array = np.array(original_values)
        tweaked_array = np.array(tweaked_values)
        
        # Calculate differences
        absolute_changes = tweaked_array - original_array
        percentage_changes = (absolute_changes / original_array * 100) if np.all(original_array != 0) else np.zeros_like(absolute_changes)
        
        return {
            'total_periods': len(original_values),
            'periods_changed': int(np.sum(absolute_changes != 0)),
            'average_absolute_change': float(np.mean(np.abs(absolute_changes))),
            'average_percentage_change': float(np.mean(percentage_changes)),
            'max_absolute_change': float(np.max(np.abs(absolute_changes))),
            'max_percentage_change': float(np.max(np.abs(percentage_changes))),
            'total_forecast_change': float(np.sum(absolute_changes)),
            'original_total': float(np.sum(original_array)),
            'tweaked_total': float(np.sum(tweaked_array)),
            'net_percentage_change': float((np.sum(tweaked_array) - np.sum(original_array)) / np.sum(original_array) * 100) if np.sum(original_array) != 0 else 0
        }
    
    def get_tweak_history(self) -> List[Dict[str, Any]]:
        """Get history of all tweaking operations"""
        return self.tweak_history.copy()
    
    def clear_tweak_history(self):
        """Clear tweaking history"""
        self.tweak_history.clear()
        self.original_forecast = None
        self.tweaked_forecast = None
        self.last_instruction = None

