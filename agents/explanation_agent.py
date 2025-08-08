"""
Agent 3: Forecast Explanation Agent
Provides conversational explanations of forecasting results from Agent 1 and Agent 2
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import numpy as np

from .base_agent import BaseAgent
from config import AGENT_CONFIG

logger = logging.getLogger(__name__)

class ExplanationAgent(BaseAgent):
    """
    Agent 3: Forecast Explanation Agent
    
    Responsibilities:
    - Provide conversational explanations of Agent 1 and Agent 2 forecasts
    - Answer specific questions about forecast variations and patterns
    - Use actual numbers and data from the forecasting process
    - Explain differences between original and tweaked forecasts
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Explanation Agent
        
        Args:
            model_path: Path to the model file
        """
        super().__init__(
            agent_name=AGENT_CONFIG["explanation_agent"]["name"],
            model_path=model_path
        )
        
        # Agent-specific configuration
        self.config = AGENT_CONFIG["explanation_agent"]
        self.max_explanation_length = self.config["max_explanation_length"]
        
        # State variables
        self.agent1_forecast = None
        self.agent2_forecast = None
        self.forecast_data = None
        self.explanation_history = []
    
    def _get_default_system_message(self) -> str:
        """Get default system message for the explanation agent"""
        return f"""You are Agent 3, a specialized Forecast Explanation Agent that provides clear, conversational explanations of forecasting results.

Your primary responsibilities:
1. Explain forecasting results from Agent 1 (Forecasting Agent) and Agent 2 (Tweaking Agent)
2. Answer specific questions about forecast patterns, variations, and methodology
3. Use actual numbers and data from the forecasting process in your explanations
4. Provide insights into why forecasts vary between different time periods
5. Explain the differences between original and tweaked forecasts
6. Make complex forecasting concepts accessible to business stakeholders

Key principles:
- Use conversational, easy-to-understand language
- Always reference specific numbers and data points
- Explain the "why" behind forecast patterns
- Compare different forecasting approaches when relevant
- Highlight key insights and business implications
- Keep explanations focused and under {self.max_explanation_length} words when possible

When explaining forecasts, consider:
- Seasonal patterns and trends in the data
- Impact of driver variables and their significance
- Model selection rationale and accuracy metrics
- Business context and practical implications
- Confidence levels and uncertainty factors

Always ground your explanations in the actual data and methodology used."""
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process explanation request
        
        Args:
            input_data: Dictionary containing:
                - question: Specific question to answer
                - agent1_forecast: Results from Agent 1 (optional)
                - agent2_forecast: Results from Agent 2 (optional)
                - forecast_data: Original data used for forecasting
                - explanation_type: Type of explanation requested
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with explanation results
        """
        try:
            logger.info(f"Processing explanation request for {self.agent_name}")
            
            # Extract input data
            question = input_data.get('question', '')
            agent1_forecast = input_data.get('agent1_forecast')
            agent2_forecast = input_data.get('agent2_forecast')
            forecast_data = input_data.get('forecast_data', {})
            explanation_type = input_data.get('explanation_type', 'general')
            
            # Store forecast data
            if agent1_forecast:
                self.agent1_forecast = agent1_forecast
            if agent2_forecast:
                self.agent2_forecast = agent2_forecast
            if forecast_data:
                self.forecast_data = forecast_data
            
            # Validate input
            if not question and explanation_type == 'question':
                return {
                    'success': False,
                    'error': 'Question is required for question-based explanations',
                    'agent': self.agent_name
                }
            
            # Generate explanation based on type
            if explanation_type == 'question' or question:
                explanation_result = self._answer_specific_question(question)
            elif explanation_type == 'comparison':
                explanation_result = self._explain_forecast_comparison()
            elif explanation_type == 'agent1':
                explanation_result = self._explain_agent1_forecast()
            elif explanation_type == 'agent2':
                explanation_result = self._explain_agent2_forecast()
            else:
                explanation_result = self._provide_general_explanation()
            
            # Prepare final result
            result = {
                'success': True,
                'agent': self.agent_name,
                'question': question,
                'explanation_type': explanation_type,
                'explanation': explanation_result['explanation'],
                'key_insights': explanation_result.get('key_insights', []),
                'data_references': explanation_result.get('data_references', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in history
            self.explanation_history.append(result)
            self.last_result = result
            
            logger.info("Explanation generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in explanation process: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _answer_specific_question(self, question: str) -> Dict[str, Any]:
        """Answer a specific question about the forecasts"""
        
        # Prepare comprehensive context
        context = self._build_forecast_context()
        
        # Create detailed prompt for AI
        prompt = f"""
        Based on the following forecasting data and results, please answer this specific question in a conversational, easy-to-understand manner:
        
        Question: "{question}"
        
        Forecasting Context:
        {context}
        
        Please provide a detailed answer that:
        1. Uses specific numbers and data points from the forecasting results
        2. Explains the underlying reasons and patterns
        3. Compares different approaches or time periods when relevant
        4. Provides business insights and implications
        5. Uses conversational language suitable for business stakeholders
        
        Keep the explanation focused and informative.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                explanation = response['text']
            else:
                explanation = self._generate_fallback_answer(question)
            
            # Extract key insights
            key_insights = self._extract_key_insights(explanation, question)
            
            return {
                'explanation': explanation,
                'key_insights': key_insights,
                'data_references': self._get_relevant_data_references(question)
            }
            
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {str(e)}")
            return {
                'explanation': self._generate_fallback_answer(question),
                'key_insights': [],
                'data_references': {}
            }
    
    def _explain_forecast_comparison(self) -> Dict[str, Any]:
        """Explain the comparison between Agent 1 and Agent 2 forecasts"""
        
        if not self.agent1_forecast or not self.agent2_forecast:
            return {
                'explanation': 'Both Agent 1 and Agent 2 forecasts are required for comparison.',
                'key_insights': [],
                'data_references': {}
            }
        
        # Calculate comparison metrics
        agent1_values = self.agent1_forecast.get('forecast_values', [])
        agent2_values = self.agent2_forecast.get('tweaked_forecast', [])
        
        if len(agent1_values) != len(agent2_values):
            return {
                'explanation': 'Cannot compare forecasts due to different lengths.',
                'key_insights': [],
                'data_references': {}
            }
        
        # Calculate differences
        differences = [a2 - a1 for a1, a2 in zip(agent1_values, agent2_values)]
        percentage_changes = [(diff / a1 * 100) if a1 != 0 else 0 for diff, a1 in zip(differences, agent1_values)]
        
        context = f"""
        Forecast Comparison Analysis:
        
        Agent 1 (Original) Forecast: {agent1_values}
        Agent 2 (Tweaked) Forecast: {agent2_values}
        Absolute Differences: {differences}
        Percentage Changes: {[f"{pc:.1f}%" for pc in percentage_changes]}
        
        Agent 1 Method: {self.agent1_forecast.get('method_used', 'Unknown')}
        Agent 1 MAPE: {self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 'Unknown')}
        
        Agent 2 Tweaking Instruction: "{self.agent2_forecast.get('human_instruction', 'Unknown')}"
        Number of Periods Adjusted: {self.agent2_forecast.get('impact_metrics', {}).get('periods_changed', 'Unknown')}
        """
        
        prompt = f"""
        Provide a conversational explanation comparing the original forecast (Agent 1) with the tweaked forecast (Agent 2):
        
        {context}
        
        Please explain:
        1. The key differences between the two forecasts
        2. Why the tweaks were made (based on the human instruction)
        3. The business impact of the changes
        4. Which forecast might be more suitable and why
        
        Use specific numbers and make it easy to understand for business stakeholders.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                explanation = response['text']
            else:
                explanation = self._generate_fallback_comparison()
            
            key_insights = [
                f"Agent 2 changed {sum(1 for d in differences if abs(d) > 0.1)} out of {len(differences)} forecast periods",
                f"Average change: {np.mean([abs(d) for d in differences]):.1f} units",
                f"Largest change: {max([abs(d) for d in differences]):.1f} units",
                f"Net total change: {sum(differences):+.1f} units"
            ]
            
            return {
                'explanation': explanation,
                'key_insights': key_insights,
                'data_references': {
                    'agent1_forecast': agent1_values,
                    'agent2_forecast': agent2_values,
                    'differences': differences
                }
            }
            
        except Exception as e:
            logger.warning(f"Comparison explanation failed: {str(e)}")
            return {
                'explanation': self._generate_fallback_comparison(),
                'key_insights': [],
                'data_references': {}
            }
    
    def _explain_agent1_forecast(self) -> Dict[str, Any]:
        """Explain Agent 1's forecasting approach and results"""
        
        if not self.agent1_forecast:
            return {
                'explanation': 'Agent 1 forecast data is not available.',
                'key_insights': [],
                'data_references': {}
            }
        
        context = self._build_agent1_context()
        
        prompt = f"""
        Provide a conversational explanation of Agent 1's forecasting approach and results:
        
        {context}
        
        Please explain:
        1. The forecasting method selected and why
        2. How the model performed (accuracy metrics)
        3. Key patterns and trends in the forecast
        4. The role of driver variables
        5. Confidence and reliability of the forecast
        
        Use specific numbers and make it accessible for business users.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                explanation = response['text']
            else:
                explanation = self._generate_fallback_agent1_explanation()
            
            return {
                'explanation': explanation,
                'key_insights': self._extract_agent1_insights(),
                'data_references': self._get_agent1_data_references()
            }
            
        except Exception as e:
            logger.warning(f"Agent 1 explanation failed: {str(e)}")
            return {
                'explanation': self._generate_fallback_agent1_explanation(),
                'key_insights': [],
                'data_references': {}
            }
    
    def _explain_agent2_forecast(self) -> Dict[str, Any]:
        """Explain Agent 2's tweaking approach and results"""
        
        if not self.agent2_forecast:
            return {
                'explanation': 'Agent 2 forecast data is not available.',
                'key_insights': [],
                'data_references': {}
            }
        
        context = self._build_agent2_context()
        
        prompt = f"""
        Provide a conversational explanation of Agent 2's forecast tweaking:
        
        {context}
        
        Please explain:
        1. How the human instruction was interpreted
        2. What specific adjustments were made
        3. The business reasoning behind the changes
        4. The impact on the overall forecast
        5. Whether the tweaks seem reasonable and justified
        
        Use specific numbers and focus on the business logic.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                explanation = response['text']
            else:
                explanation = self._generate_fallback_agent2_explanation()
            
            return {
                'explanation': explanation,
                'key_insights': self._extract_agent2_insights(),
                'data_references': self._get_agent2_data_references()
            }
            
        except Exception as e:
            logger.warning(f"Agent 2 explanation failed: {str(e)}")
            return {
                'explanation': self._generate_fallback_agent2_explanation(),
                'key_insights': [],
                'data_references': {}
            }
    
    def _provide_general_explanation(self) -> Dict[str, Any]:
        """Provide a general explanation of available forecasts"""
        
        available_forecasts = []
        if self.agent1_forecast:
            available_forecasts.append("Agent 1 (Original Forecast)")
        if self.agent2_forecast:
            available_forecasts.append("Agent 2 (Tweaked Forecast)")
        
        if not available_forecasts:
            return {
                'explanation': 'No forecast data is currently available for explanation.',
                'key_insights': [],
                'data_references': {}
            }
        
        context = self._build_forecast_context()
        
        prompt = f"""
        Provide a general overview and explanation of the available forecasting results:
        
        {context}
        
        Please provide:
        1. Summary of the forecasting process and methods used
        2. Key findings and patterns in the data
        3. Accuracy and reliability assessment
        4. Business implications and recommendations
        5. Any notable insights or concerns
        
        Make it conversational and suitable for business stakeholders.
        """
        
        try:
            if self.is_initialized:
                response = self.chat(prompt)
                explanation = response['text']
            else:
                explanation = self._generate_fallback_general_explanation()
            
            return {
                'explanation': explanation,
                'key_insights': self._extract_general_insights(),
                'data_references': self._get_general_data_references()
            }
            
        except Exception as e:
            logger.warning(f"General explanation failed: {str(e)}")
            return {
                'explanation': self._generate_fallback_general_explanation(),
                'key_insights': [],
                'data_references': {}
            }
    
    def _build_forecast_context(self) -> str:
        """Build comprehensive context about available forecasts"""
        context_parts = []
        
        if self.forecast_data:
            context_parts.append(f"Historical Data: {len(self.forecast_data.get('actuals', []))} data points")
            if 'statistics' in self.forecast_data:
                stats = self.forecast_data['statistics']
                context_parts.append(f"Historical Average: {stats.get('actuals_mean', 0):.1f}")
                context_parts.append(f"Historical Std Dev: {stats.get('actuals_std', 0):.1f}")
        
        if self.agent1_forecast:
            context_parts.append(f"Agent 1 Method: {self.agent1_forecast.get('method_used', 'Unknown')}")
            context_parts.append(f"Agent 1 MAPE: {self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 'Unknown')}")
            context_parts.append(f"Agent 1 Forecast: {self.agent1_forecast.get('forecast_values', [])}")
        
        if self.agent2_forecast:
            context_parts.append(f"Agent 2 Instruction: '{self.agent2_forecast.get('human_instruction', 'Unknown')}'")
            context_parts.append(f"Agent 2 Forecast: {self.agent2_forecast.get('tweaked_forecast', [])}")
            impact = self.agent2_forecast.get('impact_metrics', {})
            context_parts.append(f"Periods Changed: {impact.get('periods_changed', 'Unknown')}")
        
        return "\n".join(context_parts)
    
    def _build_agent1_context(self) -> str:
        """Build context specific to Agent 1"""
        if not self.agent1_forecast:
            return "Agent 1 data not available"
        
        context = f"""
        Agent 1 Forecasting Results:
        Method Used: {self.agent1_forecast.get('method_used', 'Unknown')}
        Forecast Values: {self.agent1_forecast.get('forecast_values', [])}
        MAPE: {self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 'Unknown')}%
        Correlation: {self.agent1_forecast.get('accuracy_metrics', {}).get('correlation', 'Unknown')}
        Driver Significance: {self.agent1_forecast.get('driver_significance', {}).get('is_significant', 'Unknown')}
        """
        
        if 'all_models_tested' in self.agent1_forecast:
            models = list(self.agent1_forecast['all_models_tested'].keys())
            context += f"\nModels Tested: {', '.join(models)}"
        
        return context
    
    def _build_agent2_context(self) -> str:
        """Build context specific to Agent 2"""
        if not self.agent2_forecast:
            return "Agent 2 data not available"
        
        return f"""
        Agent 2 Tweaking Results:
        Human Instruction: "{self.agent2_forecast.get('human_instruction', 'Unknown')}"
        Original Forecast: {self.agent2_forecast.get('original_forecast', {}).get('forecast_values', [])}
        Tweaked Forecast: {self.agent2_forecast.get('tweaked_forecast', [])}
        Periods Changed: {self.agent2_forecast.get('impact_metrics', {}).get('periods_changed', 'Unknown')}
        Average Change: {self.agent2_forecast.get('impact_metrics', {}).get('average_percentage_change', 'Unknown')}%
        """
    
    # Fallback explanation methods (when AI is not available)
    def _generate_fallback_answer(self, question: str) -> str:
        return f"I understand you're asking: '{question}'. Based on the available forecast data, I can provide insights about the forecasting methodology and results. However, detailed AI-powered analysis is currently unavailable."
    
    def _generate_fallback_comparison(self) -> str:
        return "The comparison between Agent 1 and Agent 2 forecasts shows the impact of human business insights on the original statistical forecast. Agent 2's tweaks reflect domain expertise and market knowledge."
    
    def _generate_fallback_agent1_explanation(self) -> str:
        if self.agent1_forecast:
            method = self.agent1_forecast.get('method_used', 'Unknown')
            mape = self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 'Unknown')
            return f"Agent 1 used {method} forecasting method, achieving a MAPE of {mape}%. The forecast is based on statistical analysis of historical patterns."
        return "Agent 1 forecast explanation is not available."
    
    def _generate_fallback_agent2_explanation(self) -> str:
        if self.agent2_forecast:
            instruction = self.agent2_forecast.get('human_instruction', 'Unknown')
            periods_changed = self.agent2_forecast.get('impact_metrics', {}).get('periods_changed', 'Unknown')
            return f"Agent 2 processed the instruction: '{instruction}' and adjusted {periods_changed} forecast periods based on business insights."
        return "Agent 2 forecast explanation is not available."
    
    def _generate_fallback_general_explanation(self) -> str:
        return "The forecasting system combines statistical methods (Agent 1) with human business insights (Agent 2) to produce accurate and business-relevant forecasts."
    
    # Helper methods for extracting insights and data references
    def _extract_key_insights(self, explanation: str, question: str) -> List[str]:
        """Extract key insights from explanation"""
        insights = []
        
        if self.agent1_forecast:
            mape = self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 0)
            if mape < 10:
                insights.append("High forecast accuracy achieved")
            elif mape > 20:
                insights.append("Forecast has higher uncertainty")
        
        if self.agent2_forecast:
            periods_changed = self.agent2_forecast.get('impact_metrics', {}).get('periods_changed', 0)
            if periods_changed > 0:
                insights.append(f"Human insights modified {periods_changed} forecast periods")
        
        return insights
    
    def _extract_agent1_insights(self) -> List[str]:
        """Extract insights specific to Agent 1"""
        insights = []
        if self.agent1_forecast:
            method = self.agent1_forecast.get('method_used', '')
            mape = self.agent1_forecast.get('accuracy_metrics', {}).get('mape', 0)
            
            insights.append(f"Used {method} forecasting method")
            insights.append(f"Achieved {mape:.1f}% MAPE")
            
            if self.agent1_forecast.get('driver_significance', {}).get('is_significant'):
                insights.append("Driver forecast is statistically significant")
            else:
                insights.append("Driver forecast not statistically significant")
        
        return insights
    
    def _extract_agent2_insights(self) -> List[str]:
        """Extract insights specific to Agent 2"""
        insights = []
        if self.agent2_forecast:
            impact = self.agent2_forecast.get('impact_metrics', {})
            insights.append(f"Modified {impact.get('periods_changed', 0)} forecast periods")
            insights.append(f"Average change: {impact.get('average_percentage_change', 0):.1f}%")
            insights.append(f"Net forecast change: {impact.get('net_percentage_change', 0):+.1f}%")
        
        return insights
    
    def _extract_general_insights(self) -> List[str]:
        """Extract general insights from all available data"""
        insights = []
        
        if self.agent1_forecast and self.agent2_forecast:
            insights.append("Both statistical and human-adjusted forecasts available")
        elif self.agent1_forecast:
            insights.append("Statistical forecast available")
        elif self.agent2_forecast:
            insights.append("Human-adjusted forecast available")
        
        return insights
    
    def _get_relevant_data_references(self, question: str) -> Dict[str, Any]:
        """Get data references relevant to the question"""
        references = {}
        
        if self.agent1_forecast:
            references['agent1_forecast'] = self.agent1_forecast.get('forecast_values', [])
            references['agent1_accuracy'] = self.agent1_forecast.get('accuracy_metrics', {})
        
        if self.agent2_forecast:
            references['agent2_forecast'] = self.agent2_forecast.get('tweaked_forecast', [])
            references['agent2_changes'] = self.agent2_forecast.get('impact_metrics', {})
        
        return references
    
    def _get_agent1_data_references(self) -> Dict[str, Any]:
        """Get data references for Agent 1"""
        if self.agent1_forecast:
            return {
                'forecast_values': self.agent1_forecast.get('forecast_values', []),
                'accuracy_metrics': self.agent1_forecast.get('accuracy_metrics', {}),
                'method_used': self.agent1_forecast.get('method_used', ''),
                'driver_significance': self.agent1_forecast.get('driver_significance', {})
            }
        return {}
    
    def _get_agent2_data_references(self) -> Dict[str, Any]:
        """Get data references for Agent 2"""
        if self.agent2_forecast:
            return {
                'tweaked_forecast': self.agent2_forecast.get('tweaked_forecast', []),
                'impact_metrics': self.agent2_forecast.get('impact_metrics', {}),
                'human_instruction': self.agent2_forecast.get('human_instruction', ''),
                'adjustments_made': self.agent2_forecast.get('tweaking_details', {}).get('adjustments_made', [])
            }
        return {}
    
    def _get_general_data_references(self) -> Dict[str, Any]:
        """Get general data references"""
        references = {}
        references.update(self._get_agent1_data_references())
        references.update(self._get_agent2_data_references())
        return references
    
    def get_explanation_history(self) -> List[Dict[str, Any]]:
        """Get history of all explanations provided"""
        return self.explanation_history.copy()
    
    def clear_explanation_history(self):
        """Clear explanation history"""
        self.explanation_history.clear()

