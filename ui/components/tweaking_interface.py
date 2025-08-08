"""
Tweaking interface component for Streamlit UI
"""
import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TweakingInterfaceComponent:
    """
    Component for Agent 2 tweaking interface
    """
    
    def __init__(self):
        """Initialize tweaking interface component"""
        self.max_instruction_length = 300
    
    def render(self, agent1_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Render tweaking interface
        
        Args:
            agent1_result: Results from Agent 1
            
        Returns:
            Dictionary with tweaking results or None
        """
        st.subheader("🔧 Forecast Tweaking")
        
        # Display current forecast summary
        self._display_forecast_summary(agent1_result)
        
        # Instruction examples
        self._display_instruction_examples()
        
        # Instruction input
        instruction = st.text_area(
            "Enter your tweaking instruction (up to 300 words):",
            height=150,
            max_chars=self.max_instruction_length * 6,  # Approximate character limit
            help="Describe how you want to adjust the forecast. Be specific about time periods and reasons."
        )
        
        # Word count
        word_count = len(instruction.split()) if instruction else 0
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if word_count > self.max_instruction_length:
                st.error(f"❌ Instruction too long: {word_count}/{self.max_instruction_length} words")
            else:
                st.info(f"Word count: {word_count}/{self.max_instruction_length}")
        
        with col2:
            if st.button("Clear", help="Clear the instruction"):
                st.rerun()
        
        # Validation and submission
        if instruction and word_count <= self.max_instruction_length:
            if st.button("🚀 Apply Tweaks", type="primary"):
                return self._process_tweaking_request(instruction, agent1_result)
        
        return None
    
    def _display_forecast_summary(self, agent1_result: Dict[str, Any]):
        """Display summary of current forecast"""
        st.markdown("### Current Forecast Summary")
        
        forecast_values = agent1_result.get('forecast_values', [])
        accuracy = agent1_result.get('accuracy_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Forecast Periods", len(forecast_values))
        with col2:
            st.metric("MAPE", f"{accuracy.get('mape', 0):.2f}%")
        with col3:
            avg_forecast = sum(forecast_values) / len(forecast_values) if forecast_values else 0
            st.metric("Avg Forecast", f"{avg_forecast:.1f}")
        with col4:
            total_forecast = sum(forecast_values) if forecast_values else 0
            st.metric("Total Forecast", f"{total_forecast:.0f}")
    
    def _display_instruction_examples(self):
        """Display example instructions"""
        with st.expander("💡 Example Instructions", expanded=False):
            st.markdown("""
            Here are some example instructions you can use:
            
            **Comparison with Drivers:**
            - "The forecast for December-25 is higher than the forecasted drivers for that month. Please adjust it to be more aligned with the driver forecast."
            
            **Comparison with Averages:**
            - "The forecast for July-25 is much lower than the average of ACD Call Volume TM1 forecast for 2025 by 35%. Please increase it accordingly."
            
            **Comparison with Previous Months:**
            - "The forecast for July-25 is much lower than last month's forecast. Please adjust it to be more consistent with the trend."
            
            **Comparison with Actuals:**
            - "The forecast for May-25 is much higher than last month's actual value. Please reduce it to be more realistic based on recent performance."
            
            **Percentage Adjustments:**
            - "Increase all forecasts for Q1 2025 by 15% due to expected market growth."
            - "Reduce the forecast for winter months (Dec-Feb) by 10% due to seasonal factors."
            
            **Business Context:**
            - "The forecast doesn't account for the new product launch in March 2025. Please increase March and April forecasts by 20%."
            """)
    
    def _process_tweaking_request(self, instruction: str, agent1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the tweaking request
        
        Args:
            instruction: Human instruction
            agent1_result: Agent 1 results
            
        Returns:
            Dictionary with tweaking results
        """
        try:
            with st.spinner("Agent 2 is processing your instruction and adjusting the forecast..."):
                # Get orchestrator from session state
                orchestrator = st.session_state.get('orchestrator')
                
                if not orchestrator:
                    st.error("❌ System not initialized. Please initialize the system first.")
                    return None
                
                # Run Agent 2 tweaking
                result = orchestrator.run_agent2_tweaking(instruction)
                
                if result['success']:
                    st.success("✅ Forecast tweaking completed!")
                    
                    # Display tweaking summary
                    self._display_tweaking_summary(result)
                    
                    return result
                else:
                    st.error(f"❌ Tweaking failed: {result.get('error', 'Unknown error')}")
                    return None
                    
        except Exception as e:
            st.error(f"❌ Error processing tweaking request: {str(e)}")
            logger.error(f"Tweaking request error: {str(e)}")
            return None
    
    def _display_tweaking_summary(self, result: Dict[str, Any]):
        """Display summary of tweaking results"""
        st.markdown("### Tweaking Results")
        
        # Impact metrics
        impact = result.get('impact_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Periods Changed", impact.get('periods_changed', 0))
        with col2:
            st.metric("Avg Change", f"{impact.get('average_percentage_change', 0):+.1f}%")
        with col3:
            st.metric("Max Change", f"{impact.get('max_percentage_change', 0):+.1f}%")
        with col4:
            st.metric("Net Change", f"{impact.get('net_percentage_change', 0):+.1f}%")
        
        # Explanation
        if 'explanation' in result:
            st.markdown("### Agent 2 Explanation")
            st.markdown(result['explanation'])
        
        # Show adjustments made
        if 'tweaking_details' in result:
            adjustments = result['tweaking_details'].get('adjustments_made', [])
            
            if adjustments and not any('error' in adj for adj in adjustments):
                with st.expander("📋 Detailed Adjustments", expanded=False):
                    for i, adj in enumerate(adjustments):
                        if 'error' not in adj:
                            st.write(f"""
                            **Period {adj.get('period_index', i) + 1}:**
                            - Original: {adj.get('original_value', 0):.1f}
                            - Adjustment: {adj.get('adjustment', 0):+.1f}
                            - New Value: {adj.get('new_value', 0):.1f}
                            - Change: {adj.get('percentage_change', 0):+.1f}%
                            """)

class TweakingGuidanceComponent:
    """
    Component for providing tweaking guidance and tips
    """
    
    def __init__(self):
        """Initialize guidance component"""
        pass
    
    def render_guidance_panel(self):
        """Render guidance panel for tweaking"""
        st.sidebar.markdown("### 🎯 Tweaking Tips")
        
        with st.sidebar.expander("📝 Writing Good Instructions"):
            st.markdown("""
            **Be Specific:**
            - Mention exact months/periods
            - Specify the comparison basis
            - Include percentage or magnitude
            
            **Use Clear Language:**
            - "Increase July-25 by 15%"
            - "Make Dec-25 lower than driver forecast"
            - "Align with historical average"
            
            **Provide Context:**
            - Explain the business reason
            - Reference market conditions
            - Mention seasonal factors
            """)
        
        with st.sidebar.expander("🔍 Common Patterns"):
            st.markdown("""
            **Seasonal Adjustments:**
            - Holiday impacts
            - Weather effects
            - Business cycles
            
            **Market Factors:**
            - Product launches
            - Competitive changes
            - Economic conditions
            
            **Operational Changes:**
            - Capacity constraints
            - Process improvements
            - Resource availability
            """)
        
        with st.sidebar.expander("⚠️ Best Practices"):
            st.markdown("""
            **Do:**
            - Be specific about time periods
            - Explain your reasoning
            - Use realistic adjustments
            - Consider business context
            
            **Don't:**
            - Make extreme changes (>50%)
            - Use vague language
            - Ignore seasonal patterns
            - Forget about constraints
            """)
    
    def render_validation_feedback(self, instruction: str) -> Dict[str, Any]:
        """
        Provide validation feedback for instruction
        
        Args:
            instruction: User instruction
            
        Returns:
            Dictionary with validation feedback
        """
        feedback = {
            'score': 0,
            'suggestions': [],
            'warnings': [],
            'strengths': []
        }
        
        if not instruction:
            return feedback
        
        instruction_lower = instruction.lower()
        words = instruction.split()
        
        # Check for specificity
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december',
                 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        has_specific_period = any(month in instruction_lower for month in months)
        if has_specific_period:
            feedback['score'] += 2
            feedback['strengths'].append("✅ Mentions specific time periods")
        else:
            feedback['suggestions'].append("💡 Consider mentioning specific months or periods")
        
        # Check for comparison basis
        comparison_terms = ['driver', 'average', 'last month', 'previous', 'actual', 'forecast']
        has_comparison = any(term in instruction_lower for term in comparison_terms)
        if has_comparison:
            feedback['score'] += 2
            feedback['strengths'].append("✅ Includes comparison basis")
        else:
            feedback['suggestions'].append("💡 Specify what to compare against (drivers, averages, etc.)")
        
        # Check for magnitude
        has_percentage = '%' in instruction or 'percent' in instruction_lower
        has_numbers = any(char.isdigit() for char in instruction)
        if has_percentage or has_numbers:
            feedback['score'] += 1
            feedback['strengths'].append("✅ Includes specific magnitude")
        else:
            feedback['suggestions'].append("💡 Consider specifying percentage or amount of change")
        
        # Check for reasoning
        reasoning_terms = ['because', 'due to', 'since', 'as', 'given', 'considering']
        has_reasoning = any(term in instruction_lower for term in reasoning_terms)
        if has_reasoning:
            feedback['score'] += 1
            feedback['strengths'].append("✅ Provides reasoning")
        else:
            feedback['suggestions'].append("💡 Explain why the adjustment is needed")
        
        # Check length
        if len(words) < 10:
            feedback['warnings'].append("⚠️ Instruction might be too brief")
        elif len(words) > 250:
            feedback['warnings'].append("⚠️ Instruction is getting long")
        
        # Normalize score
        feedback['score'] = min(10, feedback['score'])
        
        return feedback

