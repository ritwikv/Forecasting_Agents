"""
Agents module for the Agentic AI Forecasting System

This module contains the three main agents:
- Agent1: Forecasting Agent
- Agent2: Tweaking Agent  
- Agent3: Explanation Agent
"""

# Use absolute imports to avoid circular import issues
# These will be imported when needed, not at module level
__all__ = [
    'BaseAgent',
    'ForecastingAgent', 
    'TweakingAgent',
    'ExplanationAgent',
    'AgentOrchestrator'
]
