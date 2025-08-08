"""
Simple launcher that bypasses import issues
"""
import os
import sys
import subprocess

def main():
    """Launch Streamlit with minimal dependencies"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to streamlit app
        app_path = os.path.join(current_dir, "ui", "streamlit_app.py")
        
        print("🚀 Starting Agentic AI Forecasting System...")
        print("📊 Opening at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        # Set environment variable to help with imports
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        # Launch streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", app_path, 
               "--server.address", "localhost", "--server.port", "8501"]
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try installing dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()

