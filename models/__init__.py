"""
Models module for the Agentic AI Forecasting System

This module handles:
- Phi-3 model integration
- LangChain wrapper
- Model configuration and optimization
"""

from .phi3_model import Phi3Model
from .langchain_wrapper import LangChainPhi3Wrapper
from .model_config import ModelConfig

__all__ = ['Phi3Model', 'LangChainPhi3Wrapper', 'ModelConfig']

