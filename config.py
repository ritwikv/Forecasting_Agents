"""
Configuration settings for the Agentic AI Forecasting System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_path": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",  # Path to your GGUF model file
    "n_ctx": 4096,  # Context window size
    "n_threads": 8,  # Number of CPU threads (adjust based on your CPU)
    "n_gpu_layers": 0,  # 0 for CPU-only inference
    "temperature": 0.1,  # Low temperature for more deterministic outputs
    "max_tokens": 1024,  # Maximum tokens in response
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "verbose": False
}

# Agent configuration
AGENT_CONFIG = {
    "forecasting_agent": {
        "name": "Agent1_Forecasting",
        "description": "Time series forecasting agent that minimizes MAPE",
        "max_forecast_months": 24,  # Maximum months to forecast ahead
        "min_data_points": 12,  # Minimum data points required for forecasting
        "significance_threshold": 0.05,  # P-value threshold for Driver_Forecast significance
        "mape_target": 10.0  # Target MAPE percentage
    },
    "tweaking_agent": {
        "name": "Agent2_Tweaking", 
        "description": "Forecast tweaking agent based on human instructions",
        "max_instruction_length": 300,  # Maximum words in human instruction
        "adjustment_limit": 0.5  # Maximum adjustment as fraction of original forecast
    },
    "explanation_agent": {
        "name": "Agent3_Explanation",
        "description": "Forecast explanation agent providing conversational insights",
        "max_explanation_length": 500  # Maximum words in explanation
    }
}

# Data configuration
DATA_CONFIG = {
    "input_file": "Input_Data.xlsx",
    "required_columns": ["ACD Call Volume Actuals", "Driver_Forecast"],
    "target_column": "ACD_Call_Volume_Forecast",
    "date_column": "Date",  # Adjust based on your Excel file structure
    "sheet_name": 0  # Sheet index or name
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Agentic AI Forecasting System",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "forecasting_system.log"
}

# System configuration
SYSTEM_CONFIG = {
    "memory_limit_gb": 14,  # Leave 2GB for system from 16GB total
    "cpu_cores": os.cpu_count(),
    "batch_size": 32,
    "cache_size": 1000
}

