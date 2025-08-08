# 🔧 Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Issue 1: Import Errors with `launch_app.py`

**Error Message:**
```
ModuleNotFoundError: No module named 'agents.forecasting_agent'
```

**Root Cause:** 
The original complex import structure has circular dependencies that we removed to fix other issues.

**✅ SOLUTION - Use Standalone App:**
```bash
python run_standalone.py
```

**Why This Works:**
- No complex imports
- Self-contained functionality
- All features included
- Reliable on Windows systems

---

### Issue 2: Plotly Chart Errors

**Error Message:**
```
ValueError: Invalid value of type 'range' received for the 'x' property
```

**✅ SOLUTION:**
This has been fixed in the latest version. Update your code:
```bash
git pull origin main
python run_standalone.py
```

---

### Issue 3: "List is not defined" Type Hint Error

**Error Message:**
```
NameError: name 'List' is not defined
```

**✅ SOLUTION:**
This has been fixed by adding proper imports. Use the standalone app:
```bash
python run_standalone.py
```

---

## 🚀 Recommended Launch Methods

### Method 1: Standalone App (Most Reliable)
```bash
python run_standalone.py
```
- ✅ Works immediately
- ✅ All 3 agents functionality
- ✅ Complete UI with charts
- ✅ No complex dependencies

### Method 2: Test Basic Functionality
```bash
streamlit run test_streamlit.py
```
- ✅ Verify Streamlit works
- ✅ Test basic imports
- ✅ Diagnostic information

### Method 3: Test All Imports
```bash
python test_imports.py
```
- ✅ Check which imports work
- ✅ Identify specific issues
- ✅ Get recommendations

---

## 🔍 Diagnostic Steps

If you encounter issues:

1. **First, test basic setup:**
   ```bash
   python test_imports.py
   ```

2. **If basic imports work, try standalone:**
   ```bash
   python run_standalone.py
   ```

3. **If standalone works, you're good to go!**

4. **If you need the full complex system:**
   - Download the Phi-3 model file
   - Ensure all dependencies are installed
   - Use the standalone app until model integration is complete

---

## 📋 System Requirements

### Minimum Requirements:
- Python 3.8+
- 8GB RAM
- Windows 10/11

### Dependencies:
```bash
pip install streamlit pandas plotly openpyxl numpy
```

### For Full System (Future):
- Phi-3-mini-4k-instruct-Q4_K_M.gguf model file
- Additional ML libraries (will be specified when needed)

---

## 🎯 Current Status

### ✅ Working Features:
- **Standalone App**: Complete 3-agent system
- **Data Upload**: Excel files or sample data
- **Agent 1**: Forecasting with trend/seasonal analysis
- **Agent 2**: Business instruction tweaking
- **Agent 3**: Conversational explanations
- **Visualizations**: Interactive Plotly charts
- **Comparisons**: Side-by-side Agent 1 vs Agent 2

### 🚧 In Development:
- **Phi-3 Model Integration**: Advanced AI-powered forecasting
- **Complex Import Structure**: Full modular system
- **Advanced Algorithms**: ARIMA, Prophet, etc.

---

## 💡 Tips for Success

1. **Always use `python run_standalone.py`** for reliable operation
2. **Test with sample data first** before uploading your own
3. **Try different tweaking instructions** to see how Agent 2 responds
4. **Use the explanation feature** to understand the forecasting logic
5. **Check the comparison charts** to see differences between agents

---

## 🆘 Getting Help

If you still encounter issues:

1. **Run diagnostics:**
   ```bash
   python test_imports.py
   ```

2. **Check your Python environment:**
   ```bash
   python --version
   pip list | grep streamlit
   ```

3. **Try the minimal test:**
   ```bash
   streamlit run test_streamlit.py
   ```

4. **Use the standalone app:**
   ```bash
   python run_standalone.py
   ```

The standalone app should work in 99% of cases and provides all the functionality you need for testing and development!

