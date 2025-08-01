"""
Persistence manager - Read/write project state, LLM logs, and script files.

This module handles all file I/O operations for projects, including
saving analysis scripts, conversation logs, and project state.
"""

import json
import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from pathlib import Path

from .project_init import ProjectInitializer


class PersistenceManager:
    """
    Handles all project persistence and file management.
    
    This class integrates with ProjectInitializer to provide comprehensive
    project management functionality including creation, state persistence,
    and script history management.
    """
    
    def __init__(self, project_root: str = "projects"):
        self.project_root = Path(project_root)
        self.project_root.mkdir(exist_ok=True)
        self.project_initializer = ProjectInitializer(self.project_root)
    
    def create_project(
        self, 
        project_name: str, 
        dataset_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create a new project directory structure.
        
        Args:
            project_name: Name of the project (will be sanitized)
            dataset_path: Optional path to initial dataset file
            
        Returns:
            Dictionary containing project metadata and paths
            
        Raises:
            ValueError: If project name is invalid or project already exists
            FileNotFoundError: If dataset file doesn't exist
        """
        return self.project_initializer.initialize_project(project_name, dataset_path)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all available projects.
        
        Returns:
            List of project information dictionaries
        """
        return self.project_initializer.list_projects()
    
    def save_script(self, project_name: str, script_name: str, code: str):
        """
        Save an analysis script to project history.
        
        Args:
            project_name: Name of the project
            script_name: Name of the script file
            code: Python code to save
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise ValueError(f"Project '{project_name}' does not exist")
        
        history_dir = project_path / "history"
        history_dir.mkdir(exist_ok=True)
        
        # Ensure script name has .py extension
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        script_path = history_dir / script_name
        
        # If file exists, create versioned backup
        if script_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{script_path.stem}_backup_{timestamp}.py"
            backup_path = history_dir / backup_name
            script_path.rename(backup_path)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def load_script(self, project_name: str, script_name: str) -> str:
        """
        Load an analysis script from project history.
        
        Args:
            project_name: Name of the project
            script_name: Name of the script file
            
        Returns:
            Python code content of the script
            
        Raises:
            FileNotFoundError: If project or script doesn't exist
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise FileNotFoundError(f"Project '{project_name}' does not exist")
        
        # Ensure script name has .py extension
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        script_path = project_path / "history" / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script '{script_name}' not found in project '{project_name}'")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def log_llm_interaction(self, project_name: str, interaction: Dict[str, Any]):
        """
        Log LLM interaction to project log file.
        
        Args:
            project_name: Name of the project
            interaction: Dictionary containing interaction data
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise ValueError(f"Project '{project_name}' does not exist")
        
        # Find the most recent log file or create new one
        log_files = list(project_path.glob("llm_log_*.jsonl"))
        
        if log_files:
            # Use the most recent log file
            log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        else:
            # Create new log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = project_path / f"llm_log_{timestamp}.jsonl"
        
        # Add timestamp to interaction
        interaction_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **interaction
        }
        
        # Append to JSONL file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction_with_timestamp, ensure_ascii=False) + '\n')
    
    def save_project_state(self, project_name: str, state: Dict[str, Any]):
        """
        Save current project state.
        
        Args:
            project_name: Name of the project
            state: Dictionary containing project state
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise ValueError(f"Project '{project_name}' does not exist")
        
        state_file = project_path / "project_state.json"
        
        # Add metadata
        state_with_metadata = {
            "last_updated": datetime.now().isoformat(),
            "state": state
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_with_metadata, f, indent=2, ensure_ascii=False)
    
    def load_project_state(self, project_name: str) -> Dict[str, Any]:
        """
        Load project state.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing project state
            
        Raises:
            FileNotFoundError: If project or state file doesn't exist
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise FileNotFoundError(f"Project '{project_name}' does not exist")
        
        state_file = project_path / "project_state.json"
        if not state_file.exists():
            # Return empty state if no state file exists
            return {}
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # Return just the state portion, not the metadata
        return state_data.get("state", {})
    
    def get_project_info(self, project_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing project information
            
        Raises:
            FileNotFoundError: If project doesn't exist
        """
        project_path = self.project_root / project_name
        if not project_path.exists():
            raise FileNotFoundError(f"Project '{project_name}' does not exist")
        
        info = {
            "name": project_name,
            "path": str(project_path),
            "exists": True
        }
        
        # Get conventions info
        conventions_path = project_path / "conventions.json"
        if conventions_path.exists():
            try:
                with open(conventions_path, 'r', encoding='utf-8') as f:
                    conventions = json.load(f)
                    info["created"] = conventions.get("created")
                    info["has_conventions"] = True
            except (json.JSONDecodeError, IOError):
                info["has_conventions"] = False
        else:
            info["has_conventions"] = False
        
        # Count data files
        data_dir = project_path / "data"
        if data_dir.exists():
            data_files = [f for f in data_dir.iterdir() if f.is_file()]
            info["data_files"] = len(data_files)
            info["data_file_names"] = [f.name for f in data_files]
        else:
            info["data_files"] = 0
            info["data_file_names"] = []
        
        # Count history files
        history_dir = project_path / "history"
        if history_dir.exists():
            history_files = [f for f in history_dir.iterdir() if f.is_file() and f.suffix == '.py']
            info["history_files"] = len(history_files)
            info["history_file_names"] = [f.name for f in history_files]
        else:
            info["history_files"] = 0
            info["history_file_names"] = []
        
        # Count output files
        outputs_dir = project_path / "outputs"
        if outputs_dir.exists():
            output_files = [f for f in outputs_dir.iterdir() if f.is_file()]
            info["output_files"] = len(output_files)
            info["output_file_names"] = [f.name for f in output_files]
        else:
            info["output_files"] = 0
            info["output_file_names"] = []
        
        # Check for log files
        log_files = list(project_path.glob("llm_log_*.jsonl"))
        info["log_files"] = len(log_files)
        
        # Check for summary file
        summary_path = project_path / "llm_summary.md"
        info["has_summary"] = summary_path.exists()
        
        return info