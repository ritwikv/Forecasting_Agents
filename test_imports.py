"""
Test script to verify all imports work correctly
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """Test all major imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test config
        print("  ✓ Testing config...")
        from config import MODEL_CONFIG, AGENT_CONFIG, STREAMLIT_CONFIG
        print("    ✅ Config imports successful")
        
        # Test data processing
        print("  ✓ Testing data processing...")
        from data.data_processor import DataProcessor
        from data.validators import DataValidator
        print("    ✅ Data processing imports successful")
        
        # Test models
        print("  ✓ Testing models...")
        from models.phi3_model import Phi3Model
        from models.langchain_wrapper import ChatPhi3Wrapper
        print("    ✅ Model imports successful")
        
        # Test agents
        print("  ✓ Testing agents...")
        from agents.base_agent import BaseAgent
        from agents.forecasting_agent import ForecastingAgent
        from agents.tweaking_agent import TweakingAgent
        from agents.explanation_agent import ExplanationAgent
        from agents.agent_orchestrator import AgentOrchestrator
        print("    ✅ Agent imports successful")
        
        # Test utilities
        print("  ✓ Testing utilities...")
        from utils.forecasting_utils import ForecastingUtils
        from utils.statistical_tests import StatisticalTests
        from utils.state_manager import StateManager
        print("    ✅ Utility imports successful")
        
        # Test UI components
        print("  ✓ Testing UI components...")
        from ui.components.data_upload import DataUploadComponent
        from ui.components.forecast_display import ForecastDisplayComponent
        from ui.components.tweaking_interface import TweakingInterfaceComponent
        from ui.components.explanation_interface import ExplanationInterfaceComponent
        from ui.utils.chart_generator import ChartGenerator
        print("    ✅ UI component imports successful")
        
        print("\n🎉 All imports successful! The system is ready to run.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {str(e)}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality without model loading"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test data processor
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        print("  ✅ DataProcessor created successfully")
        
        # Test forecasting utils
        from utils.forecasting_utils import ForecastingUtils
        utils = ForecastingUtils()
        
        # Test with sample data
        sample_data = [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
        mape = utils.calculate_mape(sample_data[:5], sample_data[5:])
        print(f"  ✅ MAPE calculation works: {mape:.2f}%")
        
        # Test statistical tests
        from utils.statistical_tests import StatisticalTests
        stats = StatisticalTests()
        print("  ✅ StatisticalTests created successfully")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Agentic AI Forecasting System - Import Test\n")
    
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ System is ready! You can now run:")
            print("   python launch_app.py")
            print("   OR")
            print("   streamlit run ui/streamlit_app.py")
        else:
            print("\n⚠️  Imports work but functionality test failed")
    else:
        print("\n❌ Import test failed - please check dependencies")
        
    print("\n" + "="*50)

