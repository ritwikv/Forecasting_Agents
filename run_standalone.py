"""
Launch the standalone version of the forecasting system
"""
import subprocess
import sys
import os

def main():
    """Launch the standalone Streamlit app"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "standalone_app.py")
        
        print("🚀 Launching Standalone Forecasting System...")
        print("📊 This version has minimal dependencies and should work reliably")
        print("🌐 Opening at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 60)
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit pandas plotly openpyxl")

if __name__ == "__main__":
    main()

