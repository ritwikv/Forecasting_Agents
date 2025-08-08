"""
Test script to check which imports are working
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("🧪 Testing imports...")
print("-" * 50)

# Test basic imports
try:
    import streamlit as st
    print("✅ Streamlit import successful")
except Exception as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas import successful")
except Exception as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ Plotly import successful")
except Exception as e:
    print(f"❌ Plotly import failed: {e}")

print("\n🔧 Testing project imports...")
print("-" * 50)

# Test config import
try:
    from config import MODEL_CONFIG
    print("✅ Config import successful")
except Exception as e:
    print(f"❌ Config import failed: {e}")

# Test data processor import
try:
    from data.data_processor import DataProcessor
    print("✅ Data processor import successful")
except Exception as e:
    print(f"❌ Data processor import failed: {e}")

# Test utils import
try:
    from utils.forecasting_utils import ForecastingUtils
    print("✅ Forecasting utils import successful")
except Exception as e:
    print(f"❌ Forecasting utils import failed: {e}")

# Test agents import (this is where the problem likely is)
try:
    from agents.forecasting_agent import ForecastingAgent
    print("✅ Forecasting agent import successful")
except Exception as e:
    print(f"❌ Forecasting agent import failed: {e}")

try:
    from agents.agent_orchestrator import AgentOrchestrator
    print("✅ Agent orchestrator import successful")
except Exception as e:
    print(f"❌ Agent orchestrator import failed: {e}")

print("\n📋 Summary:")
print("-" * 50)
print("If you see ❌ errors above, those are the imports causing issues.")
print("The standalone app (run_standalone.py) bypasses these complex imports.")
print("\n💡 Recommendation:")
print("Use 'python run_standalone.py' for reliable operation.")
print("The original complex structure needs the Phi-3 model to be fully functional.")

