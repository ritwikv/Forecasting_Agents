# Agentic AI Forecasting System

A sophisticated multi-agent AI system for time series forecasting that combines statistical methods with human business insights.

## 🎯 Overview

This system uses three specialized AI agents working together to provide accurate, explainable, and business-relevant forecasts:

- **Agent 1 (Forecasting)**: Statistical forecasting with MAPE optimization
- **Agent 2 (Tweaking)**: Human-guided forecast adjustments  
- **Agent 3 (Explanation)**: Conversational explanations and insights

## 🚀 Features

### Agent 1: Time Series Forecasting Agent
- **MAPE Optimization**: Minimizes Mean Absolute Percentage Error
- **Statistical Significance Testing**: Tests Driver_Forecast significance automatically
- **Multiple Methods**: Exponential smoothing, linear trend, seasonal naive, ensemble
- **Smart Recommendations**: Suggests optimal forecast horizons based on data availability
- **Comprehensive Validation**: Data quality checks and model selection

### Agent 2: Forecast Tweaking Agent  
- **Natural Language Processing**: Understands business instructions up to 300 words
- **Flexible Adjustments**: Handles various comparison scenarios:
  - Comparisons with Driver_Forecast values
  - Comparisons with historical averages
  - Comparisons with previous months
  - Percentage-based adjustments
- **Business Logic**: Maintains forecast integrity while incorporating domain expertise
- **Impact Analysis**: Detailed analysis of tweaking effects

### Agent 3: Forecast Explanation Agent
- **Conversational Interface**: Easy-to-understand explanations
- **Specific Questions**: Answers targeted questions about forecasts
- **Comparative Analysis**: Explains differences between approaches
- **Data-Driven Insights**: Uses actual numbers and methodology details
- **Business Context**: Translates technical concepts for stakeholders

## 🛠️ Technical Architecture

### Core Components
- **LangChain Framework**: Agent orchestration and communication
- **Phi-3-mini Model**: Local CPU-optimized inference (4K context)
- **Streamlit UI**: Interactive web interface
- **Statistical Engine**: scikit-learn, scipy, statsmodels
- **Data Processing**: pandas, numpy for data manipulation

### System Requirements
- **RAM**: 16GB recommended
- **CPU**: Multi-core processor (optimized for CPU inference)
- **OS**: Windows 11 Pro (optimized), Linux/macOS compatible
- **Python**: 3.8+
- **Model**: Phi-3-mini-4k-instruct-Q4_K_M.gguf

## 📊 Data Requirements

### Input Format
Excel file (.xlsx/.xls) with required columns:
- `ACD Call Volume Actuals`: Historical actual values
- `Driver_Forecast`: Driver forecast values
- `Date` (optional): Date column for time indexing

### Data Quality
- Minimum 12 data points for basic forecasting
- 24+ data points recommended for seasonal patterns
- Non-negative values (call volumes)
- Consistent time intervals (monthly recommended)

## 🔧 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Forecasting_Agents
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model
Download `Phi-3-mini-4k-instruct-Q4_K_M.gguf` from:
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

Place in one of these locations:
- Project root directory
- `~/models/`
- `~/Downloads/`

### 4. Run Application
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run ui/streamlit_app.py
```

## 📖 Usage Guide

### Step 1: Initialize System
1. Configure model path in sidebar
2. Click "Initialize System"
3. Wait for agents to load

### Step 2: Load Data
1. Upload Excel file or use sample data
2. Review data validation results
3. Confirm data quality

### Step 3: Generate Forecast (Agent 1)
1. Set forecast parameters
2. Run Agent 1 forecasting
3. Review accuracy metrics and method selection

### Step 4: Tweak Forecast (Agent 2) - Optional
1. Review Agent 1 results
2. Enter business instructions (up to 300 words)
3. Apply tweaks and review changes

### Step 5: Get Explanations (Agent 3)
1. Choose explanation type
2. Ask specific questions or use quick options
3. Review AI-generated insights

### Step 6: Export Results
1. Review comprehensive results
2. Export to JSON for further analysis
3. Start new session if needed

## 💡 Example Instructions for Agent 2

### Comparison with Drivers
```
The forecast for December-25 is higher than the forecasted drivers for that month. 
Please adjust it to be more aligned with the driver forecast.
```

### Comparison with Averages
```
The forecast for July-25 is much lower than the average of ACD Call Volume TM1 
forecast for 2025 by 35%. Please increase it accordingly.
```

### Seasonal Adjustments
```
Reduce the forecast for winter months (Dec-Feb) by 10% due to seasonal factors 
and expected lower call volumes during holidays.
```

### Business Context
```
The forecast doesn't account for the new product launch in March 2025. 
Please increase March and April forecasts by 20% to reflect expected higher call volumes.
```

## 🔍 Example Questions for Agent 3

### About Agent 1
- "Why is the Agent 1 forecast much higher in Dec-25 than Sep-25?"
- "What factors contributed to the forecast accuracy?"
- "Why was this forecasting method selected over others?"

### About Agent 2
- "Why does the Agent 2 forecast vary a lot from Driver Forecast?"
- "What was the impact of the human adjustments?"
- "Are the tweaks reasonable given the business context?"

### Comparative Analysis
- "Which forecast is more reliable and why?"
- "What are the key differences between the two approaches?"
- "How should we combine both forecasts for decision making?"

## ⚙️ Configuration

### Model Configuration (`config.py`)
```python
MODEL_CONFIG = {
    "model_path": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_threads": 8,
    "temperature": 0.1,
    "max_tokens": 1024
}
```

### Agent Configuration
```python
AGENT_CONFIG = {
    "forecasting_agent": {
        "mape_target": 10.0,
        "significance_threshold": 0.05,
        "min_data_points": 12
    },
    "tweaking_agent": {
        "max_instruction_length": 300,
        "adjustment_limit": 0.5
    }
}
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading Fails**
- Verify model file path and existence
- Check available RAM (need ~3GB free)
- Ensure CPU has sufficient cores

**Data Loading Errors**
- Verify Excel file format and column names
- Check for missing or invalid data
- Ensure minimum data points requirement

**Agent Initialization Fails**
- Check Python dependencies
- Verify model file accessibility
- Review system resource availability

**Performance Issues**
- Reduce context window if memory limited
- Adjust thread count for CPU
- Close other applications to free resources

### Performance Optimization

**For Limited RAM (<12GB)**
```python
MODEL_CONFIG["n_ctx"] = 2048
MODEL_CONFIG["n_batch"] = 128
```

**For Slower CPUs**
```python
MODEL_CONFIG["n_threads"] = 4
MODEL_CONFIG["temperature"] = 0.2  # Faster inference
```

## 📈 Performance Metrics

### Typical Performance
- **Model Loading**: 10-30 seconds
- **Agent 1 Forecasting**: 30-60 seconds
- **Agent 2 Tweaking**: 15-30 seconds  
- **Agent 3 Explanation**: 20-40 seconds

### Accuracy Expectations
- **MAPE Target**: <10% for good data
- **Correlation**: >0.7 for reliable forecasts
- **Data Requirements**: 24+ points for seasonal patterns

## 🤝 Contributing

### Development Setup
1. Fork repository
2. Create feature branch
3. Install development dependencies
4. Run tests: `pytest tests/`
5. Submit pull request

### Code Structure
```
├── agents/           # AI agents implementation
├── data/            # Data processing utilities
├── models/          # Model integration
├── ui/              # Streamlit interface
├── utils/           # Utility functions
├── tests/           # Test suite
└── docs/            # Documentation
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft for Phi-3-mini model
- LangChain community for framework
- Streamlit for UI framework
- Open source scientific Python ecosystem

