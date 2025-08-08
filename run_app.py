"""
Main entry point for the Agentic AI Forecasting System
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecasting_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'scipy',
        'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if the model file exists"""
    from config import MODEL_CONFIG
    
    model_path = MODEL_CONFIG["model_path"]
    
    # Check common locations
    possible_paths = [
        model_path,
        os.path.join(os.getcwd(), model_path),
        os.path.join(os.path.expanduser('~'), 'models', model_path),
        os.path.join(os.path.expanduser('~'), 'Downloads', model_path)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Model file found at: {path}")
            return True
    
    logger.warning(f"Model file '{model_path}' not found in common locations")
    logger.warning("You can download it from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
    logger.warning("The system will still start, but you'll need to specify the correct path in the UI")
    
    return False

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'models', 'exports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def main():
    """Main entry point"""
    logger.info("Starting Agentic AI Forecasting System...")
    
    # Create necessary directories
    create_directories()
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check model file (warning only)
    check_model_file()
    
    # Import and run Streamlit app
    try:
        from ui.streamlit_app import main as run_streamlit_app
        
        logger.info("Launching Streamlit application...")
        logger.info("Access the application at: http://localhost:8501")
        
        # Run Streamlit app
        run_streamlit_app()
        
    except ImportError as e:
        logger.error(f"Failed to import Streamlit app: {str(e)}")
        logger.error("Please ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

