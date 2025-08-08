"""
Base agent class for the Agentic AI Forecasting System
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from models.langchain_wrapper import ChatPhi3Wrapper
from config import AGENT_CONFIG

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all forecasting agents
    """
    
    def __init__(self, 
                 agent_name: str,
                 model_path: Optional[str] = None,
                 system_message: Optional[str] = None):
        """
        Initialize base agent
        
        Args:
            agent_name: Name of the agent
            model_path: Path to the model file
            system_message: System message for the agent
        """
        self.agent_name = agent_name
        self.model_path = model_path
        self.system_message = system_message or self._get_default_system_message()
        
        # Initialize model wrapper
        self.model = ChatPhi3Wrapper(model_path=model_path)
        self.model.set_system_message(self.system_message)
        
        # Agent state
        self.is_initialized = False
        self.conversation_history = []
        self.last_result = None
        
        # Performance tracking
        self.execution_times = []
        self.success_count = 0
        self.error_count = 0
    
    @abstractmethod
    def _get_default_system_message(self) -> str:
        """
        Get default system message for the agent
        
        Returns:
            Default system message string
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process input data and return results
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the agent and load model
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing agent: {self.agent_name}")
            
            # Load model
            if not self.model.load_model():
                logger.error(f"Failed to load model for agent: {self.agent_name}")
                return False
            
            self.is_initialized = True
            logger.info(f"Agent {self.agent_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent {self.agent_name}: {str(e)}")
            self.is_initialized = False
            return False
    
    def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Chat with the agent
        
        Args:
            message: User message
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with chat response
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError(f"Agent {self.agent_name} not initialized")
        
        try:
            start_time = datetime.now()
            
            # Get response from model
            result = self.model.chat(message, **kwargs)
            
            # Track performance
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_times.append(execution_time)
            self.success_count += 1
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': start_time.isoformat(),
                'user_message': message,
                'agent_response': result['text'],
                'execution_time': execution_time
            })
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in agent {self.agent_name} chat: {str(e)}")
            raise
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.model.clear_conversation()
        self.conversation_history.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'agent_name': self.agent_name,
            'is_initialized': self.is_initialized,
            'total_interactions': len(self.execution_times),
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0
        }
        
        if self.execution_times:
            stats.update({
                'average_execution_time': sum(self.execution_times) / len(self.execution_times),
                'fastest_execution': min(self.execution_times),
                'slowest_execution': max(self.execution_times)
            })
        
        # Add model performance stats
        model_stats = self.model.model.get_performance_stats() if self.model.model else {}
        stats['model_performance'] = model_stats
        
        return stats
    
    def save_state(self, filepath: str):
        """
        Save agent state to file
        
        Args:
            filepath: Path to save state file
        """
        state = {
            'agent_name': self.agent_name,
            'conversation_history': self.conversation_history,
            'last_result': self.last_result,
            'performance_stats': self.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Agent state saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent state: {str(e)}")
            raise
    
    def load_state(self, filepath: str):
        """
        Load agent state from file
        
        Args:
            filepath: Path to state file
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.conversation_history = state.get('conversation_history', [])
            self.last_result = state.get('last_result')
            
            logger.info(f"Agent state loaded from: {filepath}")
        except Exception as e:
            logger.error(f"Error loading agent state: {str(e)}")
            raise
    
    def shutdown(self):
        """Shutdown the agent and cleanup resources"""
        if self.model:
            self.model.unload_model()
        self.is_initialized = False
        logger.info(f"Agent {self.agent_name} shutdown complete")
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        self.shutdown()

