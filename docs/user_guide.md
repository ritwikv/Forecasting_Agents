# User Guide: Agentic AI Forecasting System

## 🎯 Quick Start

### 1. System Initialization
1. **Launch Application**: Run `python run_app.py` or `streamlit run ui/streamlit_app.py`
2. **Configure Model**: In the sidebar, set the path to your Phi-3-mini GGUF model file
3. **Initialize System**: Click "🚀 Initialize System" and wait for agents to load
4. **Verify Status**: Check that all agents show as initialized in the sidebar

### 2. Data Upload
1. **Prepare Data**: Ensure your Excel file has columns named exactly:
   - `ACD Call Volume Actuals`
   - `Driver_Forecast`
   - `Date` (optional but recommended)
2. **Upload File**: Use the file uploader or click "📊 Load Sample Data" to try the system
3. **Review Validation**: Check data quality metrics and address any warnings
4. **Confirm Data**: Click "✅ Confirm and Use This Data" to proceed

### 3. Generate Initial Forecast (Agent 1)
1. **Set Parameters**: Choose forecast periods or specify target date
2. **Run Forecasting**: Click "🚀 Run Agent 1 Forecasting"
3. **Review Results**: Examine accuracy metrics, method selection, and forecast values
4. **Understand Significance**: Check if Driver_Forecast was deemed statistically significant

### 4. Adjust Forecast (Agent 2) - Optional
1. **Review Current Forecast**: Examine Agent 1's results and identify areas for adjustment
2. **Write Instructions**: Provide business context in natural language (up to 300 words)
3. **Apply Tweaks**: Click "🚀 Apply Tweaks" and review the changes
4. **Compare Results**: Analyze the differences between original and tweaked forecasts

### 5. Get Explanations (Agent 3)
1. **Choose Explanation Type**: Select from Agent 1, Agent 2, comparison, or custom question
2. **Ask Questions**: Use predefined questions or write your own
3. **Review Insights**: Read AI-generated explanations and key insights
4. **Follow Up**: Ask additional questions to deepen understanding

## 📊 Understanding the Interface

### Sidebar Navigation
- **System Status**: Shows initialization status and workflow progress
- **Model Configuration**: Set model path and parameters
- **Workflow Steps**: Visual progress through the forecasting process
- **Reset Session**: Clear all data and start over

### Main Content Areas

#### Data Loading Step
- **File Upload**: Drag-and-drop or browse for Excel files
- **Data Preview**: First 10 rows of your data
- **Validation Results**: Data quality checks and recommendations
- **Sample Data**: Try the system with generated sample data

#### Forecasting Step (Agent 1)
- **Parameter Settings**: Forecast periods and target dates
- **Results Display**: Accuracy metrics, method selection, and forecast values
- **Chart Visualization**: Line chart showing forecast values
- **Model Comparison**: Performance of different forecasting methods tested

#### Tweaking Step (Agent 2)
- **Current Forecast Summary**: Key metrics from Agent 1
- **Instruction Input**: Text area for business instructions
- **Example Instructions**: Common tweaking scenarios
- **Results Comparison**: Side-by-side comparison of original vs tweaked forecasts

#### Explanation Step (Agent 3)
- **Explanation Types**: Choose focus area for explanations
- **Question Interface**: Ask specific questions about forecasts
- **AI Responses**: Detailed, conversational explanations
- **Follow-up Questions**: Suggested additional questions

#### Results Step
- **Comprehensive View**: All results from all agents
- **Interactive Charts**: Dynamic visualizations with legends
- **Detailed Tables**: Numerical comparison of forecasts
- **Export Options**: Save results for further analysis

## 🔧 Advanced Features

### Custom Model Configuration
Access advanced model settings in the sidebar:
- **Context Window**: Adjust for memory constraints
- **Thread Count**: Optimize for your CPU
- **Temperature**: Control response creativity
- **Token Limits**: Manage response length

### Data Quality Optimization
- **Minimum Requirements**: 12 data points minimum, 24+ recommended
- **Missing Values**: System handles gaps with interpolation
- **Outlier Detection**: Automatic identification of extreme values
- **Correlation Analysis**: Relationship between actuals and drivers

### Forecast Customization
- **Horizon Selection**: System recommends based on data availability
- **Significance Testing**: Automatic statistical validation of drivers
- **Method Selection**: Ensemble approach for optimal accuracy
- **Confidence Intervals**: Uncertainty quantification (where applicable)

## 💡 Best Practices

### Data Preparation
1. **Consistent Intervals**: Use regular time intervals (monthly recommended)
2. **Complete Data**: Minimize missing values for better accuracy
3. **Relevant Drivers**: Ensure driver forecasts are meaningful predictors
4. **Historical Context**: Include sufficient history for pattern recognition

### Writing Effective Instructions (Agent 2)
1. **Be Specific**: Mention exact time periods and magnitudes
2. **Provide Context**: Explain the business reasoning
3. **Use Clear Language**: Avoid ambiguous terms
4. **Reference Comparisons**: Specify what to compare against

**Good Example:**
```
The forecast for December 2025 is 15% higher than the driver forecast for that month. 
Given the seasonal downturn typically seen in December, please reduce the December 
forecast to align more closely with the driver forecast, approximately a 12% reduction.
```

**Poor Example:**
```
December looks too high, make it lower.
```

### Asking Good Questions (Agent 3)
1. **Be Specific**: Ask about particular time periods or metrics
2. **Seek Understanding**: Focus on "why" and "how" questions
3. **Business Focus**: Frame questions in business terms
4. **Follow Up**: Build on previous answers for deeper insights

## 🚨 Troubleshooting

### Common Issues and Solutions

#### "Model not found" Error
- **Check Path**: Verify the model file path in sidebar settings
- **Download Model**: Get Phi-3-mini-4k-instruct-Q4_K_M.gguf from HuggingFace
- **File Permissions**: Ensure the model file is readable
- **Common Locations**: Try placing in project root, ~/models/, or ~/Downloads/

#### "Insufficient Data" Warning
- **Add More Data**: Collect additional historical data points
- **Check Quality**: Ensure data values are valid and non-negative
- **Review Columns**: Verify exact column names match requirements
- **Consider Frequency**: Monthly data works best for most use cases

#### Slow Performance
- **Reduce Context**: Lower n_ctx in model configuration
- **Fewer Threads**: Reduce n_threads if CPU is overloaded
- **Close Applications**: Free up system memory
- **Check Resources**: Monitor CPU and memory usage

#### Agent Initialization Fails
- **Dependencies**: Run `pip install -r requirements.txt`
- **Python Version**: Ensure Python 3.8 or higher
- **Memory**: Verify at least 4GB free RAM
- **Restart**: Try restarting the application

### Performance Optimization

#### For Limited Resources
```python
# In config.py, adjust these settings:
MODEL_CONFIG = {
    "n_ctx": 2048,        # Reduce context window
    "n_threads": 4,       # Fewer CPU threads
    "n_batch": 128,       # Smaller batch size
    "max_tokens": 512     # Shorter responses
}
```

#### For Better Accuracy
- **More Data**: Provide 36+ historical data points
- **Quality Drivers**: Ensure driver forecasts are well-correlated
- **Regular Updates**: Refresh forecasts with new data
- **Validation**: Compare forecasts with actual outcomes

## 📈 Interpreting Results

### Agent 1 Metrics
- **MAPE (Mean Absolute Percentage Error)**: Lower is better, <10% is excellent
- **Correlation**: Higher is better, >0.7 indicates strong relationship
- **Bias**: Close to 0 is ideal, indicates systematic over/under-forecasting
- **Method Selected**: The algorithm that performed best on your data

### Agent 2 Impact
- **Periods Changed**: Number of forecast periods modified
- **Average Change**: Mean percentage adjustment across all periods
- **Net Change**: Overall impact on total forecast
- **Business Logic**: Explanation of why changes were made

### Agent 3 Insights
- **Key Patterns**: Important trends and relationships identified
- **Confidence Levels**: Reliability assessment of forecasts
- **Business Implications**: Practical meaning of results
- **Recommendations**: Suggested actions based on analysis

## 🔄 Workflow Tips

### Iterative Improvement
1. **Start Simple**: Begin with basic forecasting
2. **Analyze Results**: Understand what the model found
3. **Add Business Context**: Use Agent 2 for domain expertise
4. **Validate Logic**: Use Agent 3 to verify reasoning
5. **Refine Iteratively**: Make incremental improvements

### Session Management
- **Save Progress**: Export results before major changes
- **Document Decisions**: Note why certain tweaks were made
- **Track Performance**: Monitor forecast accuracy over time
- **Learn from Outcomes**: Use actual results to improve future forecasts

### Collaboration
- **Share Results**: Export JSON files for team review
- **Document Assumptions**: Record business logic used in tweaks
- **Validate with Experts**: Have domain experts review explanations
- **Continuous Learning**: Incorporate feedback into future sessions

## 📞 Support and Resources

### Getting Help
- **Documentation**: Comprehensive guides in `/docs` folder
- **Examples**: Sample data and instructions provided
- **Error Messages**: Read carefully for specific guidance
- **Logs**: Check console output for detailed error information

### Community Resources
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Share experiences and best practices
- **Updates**: Check for new versions and improvements
- **Contributions**: Help improve the system

### Advanced Usage
- **API Integration**: Programmatic access to agents
- **Custom Models**: Integrate other language models
- **Extended Analytics**: Additional statistical methods
- **Automation**: Scheduled forecasting workflows

