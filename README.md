# 🤖 Agentic AI Forecasting System

A sophisticated multi-agent AI system for time series forecasting that combines statistical methods with human business insights using the LangChain framework and Phi-3-mini model.

## 🎯 Overview

This system employs three specialized AI agents working collaboratively to deliver accurate, explainable, and business-relevant forecasts:

### 🎯 **Agent 1: Forecasting Agent**
- Forecasts `ACD_Call_Volume_Forecast` using `ACD Call Volume Actuals` and `Driver_Forecast`
- Minimizes Mean Absolute Percentage Error (MAPE) through ensemble methods
- Tests `Driver_Forecast` statistical significance automatically
- Recommends optimal forecast horizons based on data availability
- Supports multiple forecasting algorithms (exponential smoothing, linear trend, seasonal naive, ensemble)

### 🔧 **Agent 2: Tweaking Agent**  
- Processes human business instructions (up to 300 words) in natural language
- Handles complex comparison scenarios:
  - Forecast vs Driver_Forecast alignment
  - Comparisons with historical averages (e.g., "35% lower than TM1 forecast average")
  - Month-to-month consistency checks
  - Seasonal and business context adjustments
- Maintains forecast integrity while incorporating domain expertise
- Provides detailed explanations of all adjustments made

### 💡 **Agent 3: Explanation Agent**
- Delivers conversational explanations of forecasting methodology and results
- Answers specific questions like "Why is Dec-25 forecast higher than Sep-25?"
- Compares Agent 1 vs Agent 2 approaches with business context
- Uses actual numbers and statistical measures in explanations
- Makes complex forecasting concepts accessible to business stakeholders

## 🚀 Key Features

- **CPU-Optimized**: Runs efficiently on 16GB RAM Windows 11 Pro systems
- **Local Inference**: Uses Phi-3-mini-4k-instruct-Q4_K_M.gguf for privacy and control
- **Interactive UI**: Streamlit-based interface with guided workflow
- **Excel Integration**: Direct import from Excel files with data validation
- **Real-time Charts**: Dynamic visualizations comparing all forecast approaches
- **Export Capabilities**: JSON export for further analysis and reporting
- **Comprehensive Testing**: Built-in statistical significance testing and validation

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent 1       │    │   Agent 2       │    │   Agent 3       │
│  Forecasting    │───▶│   Tweaking      │───▶│  Explanation    │
│                 │    │                 │    │                 │
│ • MAPE Optimize │    │ • NLP Parsing   │    │ • Conversational│
│ • Significance  │    │ • Business Logic│    │ • Q&A Interface │
│ • Multi-Method  │    │ • Adjustments   │    │ • Data-Driven   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Streamlit UI   │
                    │                 │
                    │ • Guided Flow   │
                    │ • Data Upload   │
                    │ • Visualizations│
                    │ • Export Tools  │
                    └─────────────────┘
```

## 🛠️ Technical Stack

- **AI Framework**: LangChain for agent orchestration
- **Language Model**: Phi-3-mini-4k-instruct-Q4_K_M.gguf (CPU-optimized)
- **UI Framework**: Streamlit for interactive web interface
- **Data Processing**: pandas, numpy for data manipulation
- **Statistical Analysis**: scikit-learn, scipy, statsmodels
- **Visualization**: Plotly for interactive charts
- **File Handling**: openpyxl for Excel integration

## 📋 Requirements

### System Requirements
- **RAM**: 16GB recommended (minimum 8GB)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: 5GB free space (including model file)
- **OS**: Windows 11 Pro (optimized), Linux/macOS compatible
- **Python**: 3.8 or higher

### Data Requirements
- **Format**: Excel file (.xlsx/.xls)
- **Required Columns**: 
  - `ACD Call Volume Actuals` (historical actual values)
  - `Driver_Forecast` (driver forecast values)
- **Optional**: `Date` column for time indexing
- **Minimum Data**: 12 data points (24+ recommended for seasonal patterns)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ritwikv/Forecasting_Agents.git
cd Forecasting_Agents

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Download Model

Download `Phi-3-mini-4k-instruct-Q4_K_M.gguf` from:
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

Place in project root or specify path in the UI.

### 3. Run Application

```bash
# Method 1: Simple launcher (Recommended)
python launch_app.py

# Method 2: Using run script
python run_app.py

# Method 3: Direct Streamlit
streamlit run ui/streamlit_app.py

# Method 4: Test imports first
python test_imports.py
```

### 4. Access Interface

Open your browser to: http://localhost:8501

## 📖 Usage Examples

### Agent 2 Instruction Examples

**Seasonal Adjustment:**
```
The forecast for December-25 is higher than the forecasted drivers for that month. 
Given typical holiday season patterns, please reduce the December forecast by 
approximately 15% to align with driver expectations and seasonal trends.
```

**Business Context Integration:**
```
The forecast for July-25 is much lower than the average of ACD Call Volume TM1 
forecast for 2025 by 35%. Considering the new product launch planned for Q3, 
please increase July and August forecasts by 25% to account for expected 
higher call volumes.
```

**Month-to-Month Consistency:**
```
The forecast for May-25 is much higher than last month's actual value. 
Please adjust it to show a more gradual increase of about 8% from April 
actuals to maintain realistic growth patterns.
```

### Agent 3 Question Examples

**Methodology Questions:**
- "Why is the Agent 1 forecast much higher in Dec-25 than Sep-25?"
- "What factors contributed to the selection of the ensemble method?"
- "How reliable is this forecast given the available data?"

**Comparative Analysis:**
- "Why does the Agent 2 forecast vary significantly from Driver Forecast?"
- "Which forecast approach is more suitable for our business planning?"
- "What are the key differences between the statistical and human-adjusted forecasts?"

## 📊 Sample Data

The system includes sample data generation for testing:

```python
# Sample data characteristics:
# - 36 months of historical data
# - Realistic call volume patterns with trend and seasonality
# - Correlated driver forecasts
# - Built-in noise for realistic testing
```

## 🔧 Configuration

### Model Configuration
```python
MODEL_CONFIG = {
    "model_path": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "n_ctx": 4096,          # Context window
    "n_threads": 8,         # CPU threads
    "temperature": 0.1,     # Response creativity
    "max_tokens": 1024      # Response length
}
```

### Agent Configuration
```python
AGENT_CONFIG = {
    "forecasting_agent": {
        "mape_target": 10.0,              # Target MAPE percentage
        "significance_threshold": 0.05,    # Statistical significance level
        "min_data_points": 12             # Minimum required data points
    },
    "tweaking_agent": {
        "max_instruction_length": 300,    # Maximum instruction words
        "adjustment_limit": 0.5           # Maximum adjustment ratio
    }
}
```

## 📈 Performance Metrics

### Typical Performance
- **System Initialization**: 15-45 seconds
- **Agent 1 Forecasting**: 30-90 seconds  
- **Agent 2 Tweaking**: 20-45 seconds
- **Agent 3 Explanation**: 25-60 seconds

### Accuracy Expectations
- **Target MAPE**: <10% for high-quality data
- **Correlation Threshold**: >0.7 for reliable forecasts
- **Driver Significance**: p-value <0.05 for inclusion

## 🧪 Testing

```bash
# Run basic tests
python tests/test_agents.py

# Run with pytest (if installed)
pytest tests/ -v

# Run integration test
python -c "from tests.test_agents import TestIntegration; TestIntegration().test_full_workflow_simulation()"
```

## 📁 Project Structure

```
Forecasting_Agents/
├── agents/                 # AI agents implementation
│   ├── base_agent.py      # Base agent class
│   ├── forecasting_agent.py   # Agent 1
│   ├── tweaking_agent.py      # Agent 2
│   ├── explanation_agent.py   # Agent 3
│   └── agent_orchestrator.py  # Coordination
├── data/                   # Data processing
│   ├── data_processor.py   # Excel handling
│   └── validators.py       # Data validation
├── models/                 # Model integration
│   ├── phi3_model.py      # Phi-3 wrapper
│   └── langchain_wrapper.py   # LangChain integration
├── ui/                     # Streamlit interface
│   ├── streamlit_app.py    # Main application
│   ├── components/         # UI components
│   └── utils/             # UI utilities
├── utils/                  # Utility functions
│   ├── forecasting_utils.py   # Forecasting algorithms
│   ├── statistical_tests.py   # Statistical testing
│   └── state_manager.py       # Session management
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── run_app.py            # Main entry point
└── README.md             # This file
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading Fails**
- Verify model file exists and is accessible
- Check available RAM (need ~3GB free)
- Ensure file isn't corrupted (re-download if needed)

**Performance Issues**
- Reduce `n_ctx` to 2048 for limited RAM
- Lower `n_threads` if CPU is overloaded
- Close other applications to free resources

**Data Loading Errors**
- Verify exact column names match requirements
- Check for missing or invalid data values
- Ensure minimum 12 data points available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** for the Phi-3-mini model
- **LangChain** community for the agent framework
- **Streamlit** for the excellent UI framework
- **Open Source Community** for the scientific Python ecosystem

## 📞 Support

- **Documentation**: [User Guide](docs/user_guide.md) | [API Docs](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/ritwikv/Forecasting_Agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ritwikv/Forecasting_Agents/discussions)

---

**Built with ❤️ for accurate, explainable, and business-relevant forecasting**
