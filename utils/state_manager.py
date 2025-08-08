"""
State management for the Agentic AI Forecasting System
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages state across agents and user sessions
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize state manager
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or self._generate_session_id()
        self.state = {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'agents': {},
            'data': {},
            'history': [],
            'current_step': 'initial'
        }
        self._lock = threading.Lock()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
    
    def update_agent_state(self, agent_name: str, agent_data: Dict[str, Any]):
        """
        Update state for a specific agent
        
        Args:
            agent_name: Name of the agent
            agent_data: Agent state data
        """
        with self._lock:
            self.state['agents'][agent_name] = {
                **agent_data,
                'last_updated': datetime.now().isoformat()
            }
            self.state['last_updated'] = datetime.now().isoformat()
            
            # Add to history
            self.state['history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'agent_update',
                'agent': agent_name,
                'data': agent_data
            })
    
    def get_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get state for a specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent state data or None if not found
        """
        with self._lock:
            return self.state['agents'].get(agent_name)
    
    def update_data(self, key: str, data: Any):
        """
        Update shared data
        
        Args:
            key: Data key
            data: Data to store
        """
        with self._lock:
            self.state['data'][key] = data
            self.state['last_updated'] = datetime.now().isoformat()
    
    def get_data(self, key: str) -> Any:
        """
        Get shared data
        
        Args:
            key: Data key
            
        Returns:
            Stored data or None if not found
        """
        with self._lock:
            return self.state['data'].get(key)
    
    def set_current_step(self, step: str):
        """
        Set current workflow step
        
        Args:
            step: Current step name
        """
        with self._lock:
            self.state['current_step'] = step
            self.state['last_updated'] = datetime.now().isoformat()
            
            # Add to history
            self.state['history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'step_change',
                'step': step
            })
    
    def get_current_step(self) -> str:
        """
        Get current workflow step
        
        Returns:
            Current step name
        """
        with self._lock:
            return self.state['current_step']
    
    def add_history_entry(self, action: str, data: Dict[str, Any]):
        """
        Add entry to history
        
        Args:
            action: Action type
            data: Action data
        """
        with self._lock:
            self.state['history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': action,
                **data
            })
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history entries
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of history entries
        """
        with self._lock:
            history = self.state['history']
            if limit:
                return history[-limit:]
            return history.copy()
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete state
        
        Returns:
            Complete state dictionary
        """
        with self._lock:
            return self.state.copy()
    
    def clear_state(self):
        """Clear all state data"""
        with self._lock:
            self.state = {
                'session_id': self.session_id,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'agents': {},
                'data': {},
                'history': [],
                'current_step': 'initial'
            }
    
    def save_to_file(self, filepath: str):
        """
        Save state to file
        
        Args:
            filepath: Path to save file
        """
        try:
            with self._lock:
                with open(filepath, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)
            logger.info(f"State saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise
    
    def load_from_file(self, filepath: str):
        """
        Load state from file
        
        Args:
            filepath: Path to load file
        """
        try:
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
            
            with self._lock:
                self.state = loaded_state
                self.session_id = loaded_state.get('session_id', self.session_id)
            
            logger.info(f"State loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get state summary
        
        Returns:
            State summary dictionary
        """
        with self._lock:
            return {
                'session_id': self.session_id,
                'created_at': self.state['created_at'],
                'last_updated': self.state['last_updated'],
                'current_step': self.state['current_step'],
                'agents_count': len(self.state['agents']),
                'data_keys': list(self.state['data'].keys()),
                'history_entries': len(self.state['history']),
                'active_agents': [name for name, data in self.state['agents'].items() 
                                if data.get('is_active', False)]
            }

class WorkflowManager:
    """
    Manages the workflow between agents
    """
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize workflow manager
        
        Args:
            state_manager: State manager instance
        """
        self.state_manager = state_manager
        self.workflow_steps = [
            'data_loading',
            'agent1_forecasting',
            'agent2_tweaking',
            'agent3_explanation',
            'results_display'
        ]
        self.current_step_index = 0
    
    def get_current_step(self) -> str:
        """Get current workflow step"""
        return self.state_manager.get_current_step()
    
    def advance_step(self) -> str:
        """
        Advance to next workflow step
        
        Returns:
            Next step name
        """
        current_step = self.get_current_step()
        
        if current_step in self.workflow_steps:
            current_index = self.workflow_steps.index(current_step)
            if current_index < len(self.workflow_steps) - 1:
                next_step = self.workflow_steps[current_index + 1]
                self.state_manager.set_current_step(next_step)
                return next_step
        
        return current_step
    
    def go_to_step(self, step: str) -> bool:
        """
        Go to specific workflow step
        
        Args:
            step: Target step name
            
        Returns:
            True if step is valid, False otherwise
        """
        if step in self.workflow_steps:
            self.state_manager.set_current_step(step)
            return True
        return False
    
    def reset_workflow(self):
        """Reset workflow to initial step"""
        self.state_manager.set_current_step('data_loading')
        self.current_step_index = 0
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """
        Get workflow progress information
        
        Returns:
            Progress information dictionary
        """
        current_step = self.get_current_step()
        
        if current_step in self.workflow_steps:
            current_index = self.workflow_steps.index(current_step)
            progress_percentage = (current_index / (len(self.workflow_steps) - 1)) * 100
        else:
            current_index = 0
            progress_percentage = 0
        
        return {
            'current_step': current_step,
            'current_index': current_index,
            'total_steps': len(self.workflow_steps),
            'progress_percentage': progress_percentage,
            'completed_steps': self.workflow_steps[:current_index],
            'remaining_steps': self.workflow_steps[current_index + 1:],
            'all_steps': self.workflow_steps
        }
    
    def is_step_completed(self, step: str) -> bool:
        """
        Check if a workflow step is completed
        
        Args:
            step: Step name to check
            
        Returns:
            True if step is completed, False otherwise
        """
        current_step = self.get_current_step()
        
        if step in self.workflow_steps and current_step in self.workflow_steps:
            step_index = self.workflow_steps.index(step)
            current_index = self.workflow_steps.index(current_step)
            return step_index < current_index
        
        return False
    
    def get_next_step(self) -> Optional[str]:
        """
        Get next workflow step without advancing
        
        Returns:
            Next step name or None if at end
        """
        current_step = self.get_current_step()
        
        if current_step in self.workflow_steps:
            current_index = self.workflow_steps.index(current_step)
            if current_index < len(self.workflow_steps) - 1:
                return self.workflow_steps[current_index + 1]
        
        return None
    
    def get_previous_step(self) -> Optional[str]:
        """
        Get previous workflow step
        
        Returns:
            Previous step name or None if at beginning
        """
        current_step = self.get_current_step()
        
        if current_step in self.workflow_steps:
            current_index = self.workflow_steps.index(current_step)
            if current_index > 0:
                return self.workflow_steps[current_index - 1]
        
        return None

