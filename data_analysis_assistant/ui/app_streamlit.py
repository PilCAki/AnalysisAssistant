"""
Main Streamlit application for the AnalysisAssistant.

This is the primary entry point for the chat-based data analysis interface.
Run with: streamlit run data_analysis_assistant/ui/app_streamlit.py
"""

import streamlit as st
import pandas as pd
from typing import Optional

# TODO: Import agent_core modules once implemented
# from ..agent_core.engine import AnalysisEngine
# from ..agent_core.conversation import ConversationManager
# from ..agent_core.persistence import PersistenceManager


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AnalysisAssistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🧠 AnalysisAssistant")
    st.subheader("Your AI-powered data analysis companion")
    
    # Sidebar for file upload and project management
    with st.sidebar:
        st.header("📁 Project & Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file to begin analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Loaded {uploaded_file.name}")
                
                # Show basic info
                st.write("**Dataset Info:**")
                st.write(f"- Shape: {df.shape}")
                st.write(f"- Columns: {len(df.columns)}")
                
                # Show preview
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Project selection (placeholder)
        st.header("🗂️ Projects")
        project_name = st.text_input("Project Name", placeholder="my_analysis")
        if st.button("Create/Load Project"):
            if project_name:
                st.session_state.current_project = project_name
                st.success(f"Project: {project_name}")
    
    # Main chat interface
    st.header("💬 Analysis Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display code if present
                if "code" in message:
                    st.code(message["code"], language="python")
                
                # Display results if present
                if "result" in message:
                    st.write(message["result"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response (placeholder)
        with st.chat_message("assistant"):
            if st.session_state.uploaded_data is None:
                response = "👋 Hello! Please upload a dataset first so I can help you analyze it."
            else:
                response = f"""I'd be happy to help you analyze your data! 
                
Your dataset has {st.session_state.uploaded_data.shape[0]} rows and {st.session_state.uploaded_data.shape[1]} columns.

*Note: This is a placeholder response. The full LLM integration will be implemented in future iterations.*

Here's what I can help you with:
- Data exploration and profiling
- Creating visualizations
- Statistical analysis
- Machine learning models
- Data cleaning and preprocessing

Please describe what specific analysis you'd like me to perform!"""
            
            st.write(response)
            
            # Add assistant message to session
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Instructions
    if not st.session_state.messages:
        st.info("""
        **Getting Started:**
        1. Upload your dataset using the sidebar
        2. Create or select a project
        3. Start chatting about what you want to analyze!
        
        **Example questions:**
        - "Show me a summary of this dataset"
        - "Create a plot showing the relationship between X and Y"
        - "Find correlations in the data"
        - "Build a prediction model for column Z"
        """)


if __name__ == "__main__":
    main()