# 🚨 Troubleshooting Guide

## Common Issues and Solutions

### 1. Import Error: "attempted relative import with no known parent package"

**Problem**: Getting import errors when running `streamlit run ui/streamlit_app.py`

**Solution**: Use one of these methods instead:

```bash
# Method 1: Use the simple launcher (Recommended)
python launch_app.py

# Method 2: Test imports first
python test_imports.py

# Method 3: Use the run script
python run_app.py
```

### 2. Module Not Found Errors

**Problem**: Getting "ModuleNotFoundError" for various packages

**Solution**: 
```bash
# Install all dependencies
pip install -r requirements.txt

# If using conda, create a new environment
conda create -n forecasting_agents python=3.9
conda activate forecasting_agents
pip install -r requirements.txt
```

### 3. Model Loading Issues

**Problem**: "Model file not found" or model loading errors

**Solution**:
1. Download `Phi-3-mini-4k-instruct-Q4_K_M.gguf` from HuggingFace
2. Place it in one of these locations:
   - Project root directory
   - `~/models/`
   - `~/Downloads/`
3. Or specify the full path in the Streamlit UI

### 4. Memory Issues

**Problem**: "Out of memory" or system freezing

**Solution**:
1. Close other applications
2. Reduce model parameters in `config.py`:
   ```python
   MODEL_CONFIG = {
       "n_ctx": 2048,      # Reduce from 4096
       "n_threads": 4,     # Reduce from 8
       "n_batch": 128      # Add this line
   }
   ```

### 5. Streamlit Port Issues

**Problem**: "Port 8501 is already in use"

**Solution**:
```bash
# Use a different port
streamlit run ui/streamlit_app.py --server.port 8502

# Or kill existing Streamlit processes
pkill -f streamlit
```

### 6. Excel File Issues

**Problem**: "Invalid Excel file" or column errors

**Solution**:
1. Ensure your Excel file has these exact column names:
   - `ACD Call Volume Actuals`
   - `Driver_Forecast`
2. Use the sample data option to test the system first
3. Check for missing values or non-numeric data

### 7. Performance Issues

**Problem**: System running slowly

**Solution**:
1. **For CPU optimization**:
   ```python
   # In config.py
   MODEL_CONFIG["n_threads"] = 4  # Adjust based on your CPU
   ```

2. **For memory optimization**:
   ```python
   # In config.py
   MODEL_CONFIG["n_ctx"] = 2048
   MODEL_CONFIG["max_tokens"] = 512
   ```

### 8. Agent Initialization Fails

**Problem**: "Failed to initialize agents"

**Solution**:
1. Check if model file exists and is accessible
2. Verify you have enough RAM (4GB+ free)
3. Try running `python test_imports.py` first
4. Check the console for specific error messages

## Quick Diagnostic Steps

### Step 1: Test Environment
```bash
python test_imports.py
```

### Step 2: Check Dependencies
```bash
pip list | grep -E "(streamlit|langchain|pandas|numpy|plotly)"
```

### Step 3: Test Basic Functionality
```bash
python -c "from utils.forecasting_utils import ForecastingUtils; print('✅ Basic imports work')"
```

### Step 4: Launch Application
```bash
python launch_app.py
```

## Getting Help

If you're still having issues:

1. **Check the console output** for specific error messages
2. **Run the test script** to identify the problem area
3. **Check system resources** (RAM, CPU usage)
4. **Try with sample data** first before using your own Excel file

## System Requirements Reminder

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **OS**: Windows 11 Pro (optimized), Linux/macOS compatible

## Common Error Messages

### "No module named 'langchain'"
```bash
pip install langchain==0.1.0 langchain-community==0.0.10
```

### "No module named 'streamlit'"
```bash
pip install streamlit==1.29.0
```

### "No module named 'llama_cpp'"
```bash
pip install llama-cpp-python==0.2.27
```

### "Cannot import name 'ChatPhi3Wrapper'"
This usually means the model integration needs the model file. Download the Phi-3 model first.

---

**Still having issues?** Create an issue on GitHub with:
- Your operating system
- Python version (`python --version`)
- Full error message
- Steps you've tried

