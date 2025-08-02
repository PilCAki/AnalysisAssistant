"""
UI state management and backend integration.

This module handles the connection between the UI and the backend
agent components, managing application state and data flow.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from agent_core.engine import LLMInterface
from agent_core.persistence import PersistenceManager
from agent_core.project_init import create_project_structure, sanitize_project_name, create_initial_files


@dataclass
class AppState:
    """
    Central application state management.
    """
    current_project: Optional[str] = None
    uploaded_data: Optional[pd.DataFrame] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_results: List[Dict[str, Any]] = field(default_factory=list)
    active_scripts: Dict[str, str] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    projects_directory: Path = field(default_factory=lambda: Path("projects"))


class StateManager:
    """
    Manages application state and backend integration.
    """
    
    def __init__(self, projects_dir: Optional[Path] = None):
        self.state = AppState()
        if projects_dir:
            self.state.projects_directory = projects_dir
        
        # Initialize backend components
        self.persistence = PersistenceManager()
        self.llm_interface = None  # Will be initialized when project is selected
        
        # Ensure projects directory exists
        self.state.projects_directory.mkdir(exist_ok=True)
    
    def initialize_session(self):
        """Initialize Streamlit session state."""
        if 'app_state' not in st.session_state:
            st.session_state.app_state = self.state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_project' not in st.session_state:
            st.session_state.current_project = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
    
    def list_projects(self) -> List[str]:
        """
        List available projects in the projects directory.
        
        Returns:
            List of project names
        """
        if not self.state.projects_directory.exists():
            return []
        
        projects = []
        for item in self.state.projects_directory.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                projects.append(item.name)
        
        return sorted(projects)
    
    def create_project(self, project_name: str) -> bool:
        """
        Create a new project with proper structure.
        
        Args:
            project_name: Name of the project to create
            
        Returns:
            True if project was created successfully
        """
        try:
            sanitized_name = sanitize_project_name(project_name)
            project_path = self.state.projects_directory / sanitized_name
            
            if project_path.exists():
                return False  # Project already exists
            
            create_project_structure(project_path)
            create_initial_files(project_path)
            return True
            
        except Exception as e:
            st.error(f"Error creating project: {str(e)}")
            return False
    
    def select_project(self, project_name: str):
        """
        Select and load a project.
        
        Args:
            project_name: Name of the project to select
        """
        if project_name in self.list_projects():
            self.state.current_project = project_name
            st.session_state.current_project = project_name
            
            # Initialize LLM interface for this project
            self.llm_interface = LLMInterface(project_name=project_name)
            
            # Load project data if available
            self._load_project_data(project_name)
    
    def _load_project_data(self, project_name: str):
        """Load project data and history."""
        project_path = self.state.projects_directory / project_name
        
        # Load dataset if exists
        data_dir = project_path / "data"
        if data_dir.exists():
            for data_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(data_file)
                    self.state.uploaded_data = df
                    st.session_state.uploaded_data = df
                    break  # Load first CSV found
                except Exception:
                    continue
                    
            for data_file in data_dir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(data_file)
                    self.state.uploaded_data = df
                    st.session_state.uploaded_data = df
                    break  # Load first Excel found
                except Exception:
                    continue
    
    def upload_data(self, file_data, filename: str):
        """
        Handle data upload and save to current project.
        
        Args:
            file_data: File data from Streamlit uploader
            filename: Name of the uploaded file
        """
        if not self.state.current_project:
            st.error("Please select or create a project first")
            return
        
        project_path = self.state.projects_directory / self.state.current_project
        data_dir = project_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save file to project
        file_path = data_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_data.getvalue())
        
        # Update session state
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        self.state.uploaded_data = df
        st.session_state.uploaded_data = df
    
    def process_user_message(self, message: str) -> Dict[str, Any]:
        """
        Process user message and generate response using LLM.
        
        Args:
            message: User's message/question
            
        Returns:
            Dictionary containing response, code, and results
        """
        if not self.llm_interface:
            return {
                "role": "assistant",
                "content": "Please select a project first to start analysis.",
                "code": None,
                "result": None
            }
        
        if self.state.uploaded_data is None:
            return {
                "role": "assistant", 
                "content": "Please upload a dataset first so I can help you analyze it.",
                "code": None,
                "result": None
            }
        
        try:
            # For now, provide a structured response based on the data
            # TODO: Integrate with actual LLM engine for full code generation
            df = self.state.uploaded_data
            
            response_content = f"""I'd be happy to help you analyze your data!

Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.

**Dataset Overview:**
- Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}
- Data types: {df.dtypes.value_counts().to_dict()}

**What I can help you with:**
- Data exploration and profiling
- Creating visualizations  
- Statistical analysis
- Data cleaning and preprocessing

*Full LLM integration coming soon - for now, try uploading data and exploring the interface!*
"""
            
            return {
                "role": "assistant",
                "content": response_content,
                "code": "# Data overview\nprint(f'Dataset shape: {df.shape}')\nprint('\\nColumn info:')\nprint(df.info())",
                "result": f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns"
            }
            
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}",
                "code": None,
                "result": None
            }
    
    def execute_analysis(self, code: str) -> Dict[str, Any]:
        """
        Execute analysis code and return results.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing execution results
        """
        # TODO: Implement safe code execution
        return {
            "success": True,
            "result": "Code execution not yet implemented",
            "error": None
        }
    
    def save_analysis_step(self, step_data: Dict[str, Any]):
        """
        Save analysis step to project history.
        
        Args:
            step_data: Analysis step data to save
        """
        if not self.state.current_project:
            return
            
        # Add timestamp
        step_data["timestamp"] = st.session_state.get("timestamp", "unknown")
        
        # Save to project history
        self.state.conversation_history.append(step_data)
    
    def load_project_history(self, project_name: str) -> List[Dict[str, Any]]:
        """
        Load project conversation and analysis history.
        
        Args:
            project_name: Name of the project
            
        Returns:
            List of conversation history entries
        """
        # TODO: Implement history loading from project files
        return []
    
    def export_project(self, project_name: str, format: str = "zip"):
        """
        Export project with all scripts and results.
        
        Args:
            project_name: Name of the project to export
            format: Export format (zip, tar, etc.)
        """
        # TODO: Implement project export
        pass


class UIBackendBridge:
    """
    Bridge between UI components and backend services.
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload from UI."""
        if uploaded_file is not None:
            self.state_manager.upload_data(uploaded_file, uploaded_file.name)
            return True
        return False
    
    def handle_chat_message(self, message: str) -> Dict[str, Any]:
        """Handle chat message from UI."""
        response = self.state_manager.process_user_message(message)
        self.state_manager.save_analysis_step({
            "user_message": message,
            "assistant_response": response
        })
        return response
    
    def handle_code_execution(self, code: str):
        """Handle code execution request from UI."""
        return self.state_manager.execute_analysis(code)
    
    def handle_project_operation(self, operation: str, **kwargs):
        """Handle project operations from UI."""
        if operation == "create":
            return self.state_manager.create_project(kwargs.get("name", ""))
        elif operation == "select":
            self.state_manager.select_project(kwargs.get("name", ""))
            return True
        elif operation == "list":
            return self.state_manager.list_projects()
        return False