"""
Explanation interface component for Streamlit UI
"""
import streamlit as st
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ExplanationInterfaceComponent:
    """
    Component for Agent 3 explanation interface
    """
    
    def __init__(self):
        """Initialize explanation interface component"""
        pass
    
    def render(self, forecast_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Render explanation interface
        
        Args:
            forecast_results: Results from previous agents
            
        Returns:
            Dictionary with explanation results or None
        """
        st.subheader("💡 Forecast Explanation")
        
        # Check what results are available
        available_results = self._check_available_results(forecast_results)
        
        if not available_results['has_any']:
            st.warning("⚠️ No forecast results available for explanation. Please run Agent 1 first.")
            return None
        
        # Explanation type selection
        explanation_type = self._render_explanation_type_selector(available_results)
        
        # Question input for specific questions
        question = None
        if explanation_type == 'question':
            question = self._render_question_input()
        
        # Pre-defined questions
        if explanation_type in ['agent1', 'agent2', 'comparison']:
            predefined_question = self._render_predefined_questions(explanation_type, forecast_results)
            if predefined_question:
                question = predefined_question
                explanation_type = 'question'
        
        # Generate explanation button
        if explanation_type and (explanation_type != 'question' or question):
            if st.button("🚀 Generate Explanation", type="primary"):
                return self._process_explanation_request(explanation_type, question, forecast_results)
        
        return None
    
    def _check_available_results(self, forecast_results: Dict[str, Any]) -> Dict[str, bool]:
        """Check what results are available"""
        return {
            'has_agent1': 'agent1' in forecast_results and forecast_results['agent1'],
            'has_agent2': 'agent2' in forecast_results and forecast_results['agent2'],
            'has_any': bool(forecast_results.get('agent1') or forecast_results.get('agent2'))
        }
    
    def _render_explanation_type_selector(self, available_results: Dict[str, bool]) -> str:
        """Render explanation type selector"""
        st.markdown("### Choose Explanation Type")
        
        options = []
        option_labels = {}
        
        if available_results['has_agent1']:
            options.append('agent1')
            option_labels['agent1'] = "🎯 Explain Agent 1 Forecast (Statistical Analysis)"
        
        if available_results['has_agent2']:
            options.append('agent2')
            option_labels['agent2'] = "🔧 Explain Agent 2 Tweaks (Human Adjustments)"
        
        if available_results['has_agent1'] and available_results['has_agent2']:
            options.append('comparison')
            option_labels['comparison'] = "📊 Compare Agent 1 vs Agent 2 (Side-by-side Analysis)"
        
        options.append('question')
        option_labels['question'] = "❓ Ask Specific Question (Custom Query)"
        
        # Create radio buttons
        explanation_type = st.radio(
            "Select explanation type:",
            options,
            format_func=lambda x: option_labels.get(x, x),
            key="explanation_type"
        )
        
        return explanation_type
    
    def _render_question_input(self) -> Optional[str]:
        """Render question input interface"""
        st.markdown("### Ask Your Question")
        
        # Example questions
        with st.expander("💡 Example Questions", expanded=False):
            st.markdown("""
            **About Agent 1 Forecast:**
            - "Why is the Agent 1 forecast much higher in Dec-25 than Sep-25?"
            - "What factors contributed to the forecast accuracy?"
            - "Why was this forecasting method selected over others?"
            
            **About Agent 2 Tweaks:**
            - "Why does the Agent 2 forecast vary a lot from Driver Forecast?"
            - "What was the impact of the human adjustments?"
            - "Are the tweaks reasonable given the business context?"
            
            **Comparative Questions:**
            - "Which forecast is more reliable and why?"
            - "What are the key differences between the two approaches?"
            - "How do the forecasts compare to historical patterns?"
            
            **Technical Questions:**
            - "What is the confidence level of these forecasts?"
            - "How sensitive are the forecasts to input changes?"
            - "What are the main limitations of each approach?"
            """)
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask anything about the forecasts, methodology, or results...",
            help="Be specific about what you want to understand"
        )
        
        return question.strip() if question else None
    
    def _render_predefined_questions(self, explanation_type: str, forecast_results: Dict[str, Any]) -> Optional[str]:
        """Render predefined questions for quick access"""
        st.markdown("### Quick Questions")
        
        questions = []
        
        if explanation_type == 'agent1':
            questions = [
                "Why was this forecasting method selected?",
                "How accurate is this forecast?",
                "What role did the driver forecast play?",
                "What are the key patterns in the forecast?",
                "What factors influence the forecast reliability?"
            ]
        elif explanation_type == 'agent2':
            questions = [
                "How were the human instructions interpreted?",
                "What specific adjustments were made?",
                "Why were these tweaks necessary?",
                "What is the business impact of the changes?",
                "Are the adjustments reasonable?"
            ]
        elif explanation_type == 'comparison':
            questions = [
                "What are the key differences between the forecasts?",
                "Which forecast is more suitable for business use?",
                "How do the approaches complement each other?",
                "What are the trade-offs between statistical and human-adjusted forecasts?",
                "Which forecast has higher confidence?"
            ]
        
        if questions:
            selected_question = st.selectbox(
                "Choose a quick question:",
                [""] + questions,
                key=f"quick_question_{explanation_type}"
            )
            
            return selected_question if selected_question else None
        
        return None
    
    def _process_explanation_request(self, 
                                   explanation_type: str, 
                                   question: Optional[str],
                                   forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process explanation request
        
        Args:
            explanation_type: Type of explanation
            question: Specific question (if any)
            forecast_results: Available forecast results
            
        Returns:
            Dictionary with explanation results
        """
        try:
            with st.spinner("Agent 3 is analyzing the forecasts and generating explanation..."):
                # Get orchestrator from session state
                orchestrator = st.session_state.get('orchestrator')
                
                if not orchestrator:
                    st.error("❌ System not initialized. Please initialize the system first.")
                    return None
                
                # Run Agent 3 explanation
                result = orchestrator.run_agent3_explanation(
                    question=question,
                    explanation_type=explanation_type
                )
                
                if result['success']:
                    st.success("✅ Explanation generated!")
                    
                    # Display explanation
                    self._display_explanation_result(result)
                    
                    return result
                else:
                    st.error(f"❌ Explanation generation failed: {result.get('error', 'Unknown error')}")
                    return None
                    
        except Exception as e:
            st.error(f"❌ Error processing explanation request: {str(e)}")
            logger.error(f"Explanation request error: {str(e)}")
            return None
    
    def _display_explanation_result(self, result: Dict[str, Any]):
        """Display explanation result"""
        st.markdown("### 🤖 AI Explanation")
        
        # Main explanation
        explanation = result.get('explanation', 'No explanation available')
        st.markdown(explanation)
        
        # Key insights
        if 'key_insights' in result and result['key_insights']:
            st.markdown("### 🔍 Key Insights")
            for insight in result['key_insights']:
                st.write(f"• {insight}")
        
        # Data references
        if 'data_references' in result and result['data_references']:
            with st.expander("📊 Data References", expanded=False):
                data_refs = result['data_references']
                
                for key, value in data_refs.items():
                    if isinstance(value, list) and value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value[:5]}{'...' if len(value) > 5 else ''}")
                    elif isinstance(value, dict) and value:
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                st.write(f"  - {sub_key}: {sub_value:.3f}")
                            else:
                                st.write(f"  - {sub_key}: {sub_value}")
        
        # Follow-up questions
        self._render_followup_questions(result)
    
    def _render_followup_questions(self, result: Dict[str, Any]):
        """Render suggested follow-up questions"""
        st.markdown("### 🔄 Follow-up Questions")
        
        explanation_type = result.get('explanation_type', 'general')
        
        followup_questions = []
        
        if explanation_type == 'agent1':
            followup_questions = [
                "How can we improve the forecast accuracy?",
                "What would happen if we had more historical data?",
                "How sensitive is the forecast to recent trends?",
                "What are the main uncertainty factors?"
            ]
        elif explanation_type == 'agent2':
            followup_questions = [
                "What other adjustments could be considered?",
                "How do these tweaks affect forecast uncertainty?",
                "What business scenarios support these changes?",
                "How often should forecasts be tweaked?"
            ]
        elif explanation_type == 'comparison':
            followup_questions = [
                "How should we combine both forecasts?",
                "What decision criteria should we use?",
                "How do we validate the better approach?",
                "What are the implementation considerations?"
            ]
        else:
            followup_questions = [
                "Can you explain the methodology in more detail?",
                "What are the business implications?",
                "How confident should we be in these results?",
                "What additional analysis would be helpful?"
            ]
        
        # Display as buttons
        cols = st.columns(2)
        for i, question in enumerate(followup_questions):
            with cols[i % 2]:
                if st.button(question, key=f"followup_{i}"):
                    # Store the question for next interaction
                    st.session_state.followup_question = question
                    st.rerun()

class ExplanationHistoryComponent:
    """
    Component for managing explanation history
    """
    
    def __init__(self):
        """Initialize explanation history component"""
        pass
    
    def render_history_sidebar(self):
        """Render explanation history in sidebar"""
        if 'explanation_history' not in st.session_state:
            st.session_state.explanation_history = []
        
        st.sidebar.markdown("### 📚 Explanation History")
        
        history = st.session_state.explanation_history
        
        if not history:
            st.sidebar.write("No explanations yet")
            return
        
        # Display recent explanations
        for i, explanation in enumerate(reversed(history[-5:])):  # Show last 5
            with st.sidebar.expander(f"Explanation {len(history) - i}", expanded=False):
                st.write(f"**Type:** {explanation.get('explanation_type', 'Unknown')}")
                if explanation.get('question'):
                    st.write(f"**Question:** {explanation['question'][:50]}...")
                st.write(f"**Time:** {explanation.get('timestamp', 'Unknown')[:19]}")
                
                if st.button(f"View Full", key=f"view_explanation_{i}"):
                    st.session_state.selected_explanation = explanation
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.explanation_history = []
            st.rerun()
    
    def add_to_history(self, explanation_result: Dict[str, Any]):
        """Add explanation to history"""
        if 'explanation_history' not in st.session_state:
            st.session_state.explanation_history = []
        
        st.session_state.explanation_history.append(explanation_result)
        
        # Keep only last 20 explanations
        if len(st.session_state.explanation_history) > 20:
            st.session_state.explanation_history = st.session_state.explanation_history[-20:]

