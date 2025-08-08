"""
Simple launcher for the Agentic AI Forecasting System
This script can be run directly to launch the Streamlit app
"""
import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Launch the Streamlit application"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the Streamlit app
        streamlit_app_path = os.path.join(script_dir, "ui", "streamlit_app.py")
        
        # Check if the Streamlit app exists
        if not os.path.exists(streamlit_app_path):
            logger.error(f"Streamlit app not found at: {streamlit_app_path}")
            return
        
        logger.info("🚀 Launching Agentic AI Forecasting System...")
        logger.info("📊 Access the application at: http://localhost:8501")
        logger.info("⏹️  Press Ctrl+C to stop the application")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            streamlit_app_path,
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Error launching application: {str(e)}")
        logger.error("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

