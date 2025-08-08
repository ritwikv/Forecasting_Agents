"""
Agent Orchestrator for coordinating the three forecasting agents
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .forecasting_agent import ForecastingAgent
from .tweaking_agent import TweakingAgent
from .explanation_agent import ExplanationAgent
from utils.state_manager import StateManager, WorkflowManager
from data.data_processor import DataProcessor
from config import AGENT_CONFIG

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates the interaction between the three forecasting agents
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Agent Orchestrator
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        
        # Initialize state management
        self.state_manager = StateManager()
        self.workflow_manager = WorkflowManager(self.state_manager)
        
        # Initialize agents
        self.agents = {}
        self.agent_instances = {}
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Execution tracking
        self.execution_history = []
        self.is_initialized = False
    
    def initialize_agents(self) -> bool:
        """
        Initialize all agents
        
        Returns:
            True if all agents initialized successfully, False otherwise
        """
        try:
            logger.info("Initializing agents...")
            
            # Initialize Agent 1 (Forecasting)
            self.agent_instances['agent1'] = ForecastingAgent(self.model_path)
            if not self.agent_instances['agent1'].initialize():
                logger.error("Failed to initialize Agent 1")
                return False
            
            # Initialize Agent 2 (Tweaking)
            self.agent_instances['agent2'] = TweakingAgent(self.model_path)
            if not self.agent_instances['agent2'].initialize():
                logger.error("Failed to initialize Agent 2")
                return False
            
            # Initialize Agent 3 (Explanation)
            self.agent_instances['agent3'] = ExplanationAgent(self.model_path)
            if not self.agent_instances['agent3'].initialize():
                logger.error("Failed to initialize Agent 3")
                return False
            
            self.is_initialized = True
            logger.info("All agents initialized successfully")
            
            # Update state
            self.state_manager.set_current_step('data_loading')
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            return False
    
    def load_data(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and process data for forecasting
        
        Args:
            file_path: Optional path to Excel file
            
        Returns:
            Dictionary with data loading results
        """
        try:
            logger.info("Loading and processing data...")
            
            # Set file path if provided
            if file_path:
                self.data_processor.file_path = file_path
            
            # Load and process data
            raw_data = self.data_processor.load_data()
            processed_data = self.data_processor.preprocess_data()
            forecast_data = self.data_processor.get_forecast_data()
            data_summary = self.data_processor.get_data_summary()
            
            # Store in state
            self.state_manager.update_data('raw_data', raw_data.to_dict())
            self.state_manager.update_data('processed_data', processed_data.to_dict())
            self.state_manager.update_data('forecast_data', forecast_data)
            self.state_manager.update_data('data_summary', data_summary)
            
            # Advance workflow
            self.workflow_manager.advance_step()
            
            result = {
                'success': True,
                'data_summary': data_summary,
                'forecast_data': forecast_data,
                'message': f"Successfully loaded {len(processed_data)} data points"
            }
            
            logger.info("Data loading completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to load data'
            }
    
    def run_agent1_forecast(self, 
                           forecast_periods: Optional[int] = None,
                           target_year: Optional[int] = None,
                           target_month: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Agent 1 forecasting
        
        Args:
            forecast_periods: Number of periods to forecast
            target_year: Target year for forecast
            target_month: Target month for forecast
            
        Returns:
            Dictionary with forecasting results
        """
        try:
            if not self.is_initialized:
                if not self.initialize_agents():
                    return {
                        'success': False,
                        'error': 'Failed to initialize agents',
                        'agent': 'agent1'
                    }
            
            logger.info("Running Agent 1 forecasting...")
            
            # Get forecast data from state
            forecast_data = self.state_manager.get_data('forecast_data')
            if not forecast_data:
                return {
                    'success': False,
                    'error': 'No forecast data available. Please load data first.',
                    'agent': 'agent1'
                }
            
            # Prepare input for Agent 1
            input_data = {
                'actuals': forecast_data['actuals'],
                'driver_forecast': forecast_data['driver_forecast'],
                'forecast_periods': forecast_periods,
                'target_year': target_year,
                'target_month': target_month
            }
            
            # Run Agent 1
            result = self.agent_instances['agent1'].process(input_data)
            
            if result['success']:
                # Store result in state
                self.state_manager.update_agent_state('agent1', result)
                
                # Advance workflow
                self.workflow_manager.advance_step()
                
                logger.info("Agent 1 forecasting completed successfully")
            else:
                logger.error(f"Agent 1 forecasting failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Agent 1 forecasting: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': 'agent1'
            }
    
    def run_agent2_tweaking(self, human_instruction: str) -> Dict[str, Any]:
        """
        Run Agent 2 forecast tweaking
        
        Args:
            human_instruction: Human instruction for tweaking
            
        Returns:
            Dictionary with tweaking results
        """
        try:
            if not self.is_initialized:
                if not self.initialize_agents():
                    return {
                        'success': False,
                        'error': 'Failed to initialize agents',
                        'agent': 'agent2'
                    }
            
            logger.info("Running Agent 2 tweaking...")
            
            # Get Agent 1 results from state
            agent1_result = self.state_manager.get_agent_state('agent1')
            if not agent1_result or not agent1_result.get('success'):
                return {
                    'success': False,
                    'error': 'Agent 1 forecast is required before tweaking. Please run Agent 1 first.',
                    'agent': 'agent2'
                }
            
            # Get forecast data from state
            forecast_data = self.state_manager.get_data('forecast_data')
            
            # Prepare input for Agent 2
            input_data = {
                'original_forecast': agent1_result,
                'human_instruction': human_instruction,
                'forecast_data': forecast_data
            }
            
            # Run Agent 2
            result = self.agent_instances['agent2'].process(input_data)
            
            if result['success']:
                # Store result in state
                self.state_manager.update_agent_state('agent2', result)
                
                # Advance workflow
                self.workflow_manager.advance_step()
                
                logger.info("Agent 2 tweaking completed successfully")
            else:
                logger.error(f"Agent 2 tweaking failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Agent 2 tweaking: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': 'agent2'
            }
    
    def run_agent3_explanation(self, 
                              question: Optional[str] = None,
                              explanation_type: str = 'general') -> Dict[str, Any]:
        """
        Run Agent 3 explanation
        
        Args:
            question: Specific question to answer
            explanation_type: Type of explanation ('general', 'agent1', 'agent2', 'comparison', 'question')
            
        Returns:
            Dictionary with explanation results
        """
        try:
            if not self.is_initialized:
                if not self.initialize_agents():
                    return {
                        'success': False,
                        'error': 'Failed to initialize agents',
                        'agent': 'agent3'
                    }
            
            logger.info(f"Running Agent 3 explanation (type: {explanation_type})...")
            
            # Get results from state
            agent1_result = self.state_manager.get_agent_state('agent1')
            agent2_result = self.state_manager.get_agent_state('agent2')
            forecast_data = self.state_manager.get_data('forecast_data')
            
            # Prepare input for Agent 3
            input_data = {
                'question': question,
                'explanation_type': explanation_type,
                'agent1_forecast': agent1_result,
                'agent2_forecast': agent2_result,
                'forecast_data': forecast_data
            }
            
            # Run Agent 3
            result = self.agent_instances['agent3'].process(input_data)
            
            if result['success']:
                # Store result in state
                self.state_manager.update_agent_state('agent3', result)
                
                # Advance workflow if not already at final step
                if self.workflow_manager.get_current_step() != 'results_display':
                    self.workflow_manager.advance_step()
                
                logger.info("Agent 3 explanation completed successfully")
            else:
                logger.error(f"Agent 3 explanation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Agent 3 explanation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': 'agent3'
            }
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        Get results from all agents
        
        Returns:
            Dictionary with all agent results
        """
        return {
            'agent1_result': self.state_manager.get_agent_state('agent1'),
            'agent2_result': self.state_manager.get_agent_state('agent2'),
            'agent3_result': self.state_manager.get_agent_state('agent3'),
            'forecast_data': self.state_manager.get_data('forecast_data'),
            'data_summary': self.state_manager.get_data('data_summary'),
            'workflow_progress': self.workflow_manager.get_workflow_progress(),
            'session_summary': self.state_manager.get_summary()
        }
    
    def reset_session(self):
        """Reset the entire session"""
        logger.info("Resetting session...")
        
        # Clear state
        self.state_manager.clear_state()
        self.workflow_manager.reset_workflow()
        
        # Clear agent histories
        for agent in self.agent_instances.values():
            if hasattr(agent, 'clear_conversation'):
                agent.clear_conversation()
        
        # Clear execution history
        self.execution_history.clear()
        
        logger.info("Session reset completed")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status
        
        Returns:
            Dictionary with workflow status
        """
        return {
            'current_step': self.workflow_manager.get_current_step(),
            'progress': self.workflow_manager.get_workflow_progress(),
            'agents_initialized': self.is_initialized,
            'available_actions': self._get_available_actions(),
            'session_summary': self.state_manager.get_summary()
        }
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions based on current workflow step"""
        current_step = self.workflow_manager.get_current_step()
        
        actions = []
        
        if current_step == 'data_loading':
            actions.append('load_data')
        elif current_step == 'agent1_forecasting':
            actions.append('run_agent1_forecast')
        elif current_step == 'agent2_tweaking':
            actions.extend(['run_agent2_tweaking', 'skip_tweaking'])
        elif current_step == 'agent3_explanation':
            actions.append('run_agent3_explanation')
        elif current_step == 'results_display':
            actions.extend(['get_all_results', 'export_results', 'reset_session'])
        
        # Always available actions
        actions.extend(['get_workflow_status', 'get_agent_performance'])
        
        return actions
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """
        Get performance statistics for all agents
        
        Returns:
            Dictionary with agent performance metrics
        """
        performance = {}
        
        for agent_name, agent_instance in self.agent_instances.items():
            if hasattr(agent_instance, 'get_performance_stats'):
                performance[agent_name] = agent_instance.get_performance_stats()
        
        return performance
    
    def export_results(self, filepath: str) -> Dict[str, Any]:
        """
        Export all results to file
        
        Args:
            filepath: Path to export file
            
        Returns:
            Dictionary with export status
        """
        try:
            results = self.get_all_results()
            
            # Add metadata
            results['export_metadata'] = {
                'exported_at': datetime.now().isoformat(),
                'session_id': self.state_manager.session_id,
                'workflow_completed': self.workflow_manager.get_current_step() == 'results_display'
            }
            
            # Save to file
            self.state_manager.save_to_file(filepath)
            
            return {
                'success': True,
                'filepath': filepath,
                'message': 'Results exported successfully'
            }
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to export results'
            }
    
    def shutdown(self):
        """Shutdown all agents and cleanup resources"""
        logger.info("Shutting down orchestrator...")
        
        for agent_instance in self.agent_instances.values():
            if hasattr(agent_instance, 'shutdown'):
                agent_instance.shutdown()
        
        self.agent_instances.clear()
        self.is_initialized = False
        
        logger.info("Orchestrator shutdown completed")
    
    def __del__(self):
        """Cleanup when orchestrator is destroyed"""
        self.shutdown()

