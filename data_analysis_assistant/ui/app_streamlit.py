"""
Main Streamlit application for the AnalysisAssistant.

This is the primary entry point for the chat-based data analysis interface.
Run with: streamlit run data_analysis_assistant/ui/app_streamlit.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional
import sys
import os

# Add the parent directory to sys.path to allow imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ui.state import StateManager, UIBackendBridge
from ui.components import (
    ChatComponents, DataComponents, ProjectComponents, SettingsComponents
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'state_manager' not in st.session_state:
        # Get the absolute path to projects directory
        projects_dir = Path(__file__).parent.parent.parent / "projects"
        st.session_state.state_manager = StateManager(projects_dir)
        st.session_state.ui_bridge = UIBackendBridge(st.session_state.state_manager)
    
    st.session_state.state_manager.initialize_session()


def handle_project_management():
    """Handle project selection and creation."""
    state_manager = st.session_state.state_manager
    ui_bridge = st.session_state.ui_bridge
    
    # Get list of available projects
    projects = ui_bridge.handle_project_operation("list")
    
    # Project selector component
    project_action = ProjectComponents.project_selector(
        projects, 
        st.session_state.get('current_project')
    )
    
    if project_action:
        if project_action.startswith("CREATE:"):
            # Create new project
            project_name = project_action.split(":", 1)[1]
            success = ui_bridge.handle_project_operation("create", name=project_name)
            if success:
                st.success(f"Created project: {project_name}")
                # Select the newly created project
                ui_bridge.handle_project_operation("select", name=project_name)
                st.rerun()
            else:
                st.error("Failed to create project or project already exists")
        else:
            # Select existing project
            ui_bridge.handle_project_operation("select", name=project_action)
            st.success(f"Selected project: {project_action}")
            st.rerun()
    
    # Display current project info
    ProjectComponents.project_info_display(
        st.session_state.get('current_project')
    )


def handle_file_upload():
    """Handle file upload functionality."""
    ui_bridge = st.session_state.ui_bridge
    
    st.subheader("📤 Dataset Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'tsv', 'xlsx', 'xls'],
        help="Upload a CSV, TSV, or Excel file to begin analysis"
    )
    
    if uploaded_file is not None:
        ui_bridge.handle_file_upload(uploaded_file)


def display_chat_history():
    """Display the chat conversation history."""
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            ChatComponents.display_message(
                role=message["role"],
                content=message["content"],
                code=message.get("code"),
                result=message.get("result")
            )


def handle_chat_input():
    """Handle new chat messages from user."""
    ui_bridge = st.session_state.ui_bridge
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        ChatComponents.display_message(
            role="user",
            content=prompt
        )
        
        # Get assistant response
        with st.spinner("Thinking..."):
            response = ui_bridge.handle_chat_message(prompt)
        
        # Add assistant response to chat
        st.session_state.messages.append(response)
        
        # Display assistant response
        ChatComponents.display_message(
            role=response["role"],
            content=response["content"],
            code=response.get("code"),
            result=response.get("result")
        )
        
        # Rerun to update the display
        st.rerun()


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
    
    # Sidebar for project management and file upload
    with st.sidebar:
        st.header("🎯 Project & Data Management")
        
        # Project management
        handle_project_management()
        
        st.divider()
        
        # File upload
        handle_file_upload()
        
        st.divider()
        
        # Settings
        SettingsComponents.model_settings()
        SettingsComponents.analysis_preferences()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Analysis Chat")
    
    # Display chat history
    display_chat_history()
    
    # Handle new chat input
    handle_chat_input()
    
    # Instructions for new users
    if not st.session_state.messages:
        st.info("""
        **Getting Started:**
        1. 📁 Create or select a project in the sidebar
        2. 📤 Upload your dataset (CSV or Excel file)
        3. 💬 Start chatting about what you want to analyze!
        
        **Example questions:**
        - "Show me a summary of this dataset"
        - "Create a plot showing the relationship between X and Y"
        - "Find correlations in the data"
        - "What insights can you find in this data?"
        """)


if __name__ == "__main__":
    main()