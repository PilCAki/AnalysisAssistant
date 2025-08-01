"""
Project initialization module for AnalysisAssistant.

This module handles the creation of new analysis project workspaces,
including directory structure setup, conventions scaffolding, and file management.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name for filesystem compatibility.
    
    Args:
        name: Raw project name from user input
        
    Returns:
        Sanitized project name safe for filesystem use
    """
    # Convert to lowercase and replace spaces with underscores
    sanitized = name.lower().strip()
    
    # Replace spaces and multiple whitespace with single underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # Remove or replace invalid filesystem characters
    # Keep only alphanumeric, underscores, hyphens, and dots
    sanitized = re.sub(r'[^a-z0-9_\-\.]', '', sanitized)
    
    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip('._')
    
    # Ensure the name is not empty after sanitization
    if not sanitized:
        raise ValueError("Project name cannot be empty after sanitization")
    
    # Limit length to reasonable filesystem limits
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip('_')
    
    return sanitized


def create_conventions_scaffold() -> Dict[str, Any]:
    """
    Create the initial conventions.json scaffold structure.
    
    Returns:
        Dictionary containing the initial conventions structure
    """
    return {
        "created": datetime.now().isoformat(),
        "columns": {},
        "notes": "",
        "model_assumptions": {},
        "global_settings": {}
    }


def create_project_structure(project_path: Path) -> None:
    """
    Create the standardized project directory structure.
    
    Args:
        project_path: Path to the project directory
    """
    # Create main project directory
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["data", "history", "outputs"]
    for subdir in subdirs:
        (project_path / subdir).mkdir(exist_ok=True)


def create_initial_files(project_path: Path) -> None:
    """
    Create initial project files (conventions.json, logs, summary).
    
    Args:
        project_path: Path to the project directory
    """
    # Create conventions.json if it doesn't exist
    conventions_path = project_path / "conventions.json"
    if not conventions_path.exists():
        conventions_data = create_conventions_scaffold()
        with open(conventions_path, 'w', encoding='utf-8') as f:
            json.dump(conventions_data, f, indent=2, ensure_ascii=False)
    
    # Create initial LLM log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = project_path / f"llm_log_{timestamp}.jsonl"
    if not log_path.exists():
        # Create empty JSONL file
        log_path.touch()
    
    # Create initial summary markdown file
    summary_path = project_path / "llm_summary.md"
    if not summary_path.exists():
        summary_content = f"""# Analysis Summary

Project created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Current Session Context

*No analysis sessions yet.*

## Key Findings

*Analysis findings will be summarized here.*

## Next Steps

*Planned analysis steps will be listed here.*
"""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)


def copy_dataset_to_project(dataset_path: Union[str, Path], project_path: Path) -> Path:
    """
    Copy the uploaded dataset to the project's data directory.
    
    Args:
        dataset_path: Path to the source dataset file
        project_path: Path to the project directory
        
    Returns:
        Path to the copied dataset file in the project
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset file extension is not supported
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Check if file extension is supported
    supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv'}
    if dataset_path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported dataset format: {dataset_path.suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )
    
    # Copy to data directory
    data_dir = project_path / "data"
    destination_path = data_dir / dataset_path.name
    
    # If file already exists, create a backup or versioned name
    if destination_path.exists():
        counter = 1
        stem = dataset_path.stem
        suffix = dataset_path.suffix
        while destination_path.exists():
            destination_path = data_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    
    shutil.copy2(dataset_path, destination_path)
    return destination_path


def load_existing_conventions(project_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load existing conventions.json if it exists.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dictionary containing conventions data, or None if file doesn't exist
    """
    conventions_path = project_path / "conventions.json"
    
    if not conventions_path.exists():
        return None
    
    try:
        with open(conventions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # Log error but don't fail - we'll create a new conventions file
        print(f"Warning: Could not read existing conventions.json: {e}")
        return None


class ProjectInitializer:
    """
    Main class for initializing new analysis projects.
    """
    
    def __init__(self, projects_root: Union[str, Path] = "projects"):
        """
        Initialize the project initializer.
        
        Args:
            projects_root: Root directory for all projects
        """
        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(exist_ok=True)
    
    def initialize_project(
        self, 
        project_name: str, 
        dataset_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Initialize a new analysis project workspace.
        
        Args:
            project_name: Name of the project (will be sanitized)
            dataset_path: Optional path to initial dataset file
            
        Returns:
            Dictionary containing project metadata and paths
            
        Raises:
            ValueError: If project name is invalid or project already exists
            FileNotFoundError: If dataset file doesn't exist
        """
        # Sanitize project name
        sanitized_name = sanitize_project_name(project_name)
        
        if not sanitized_name:
            raise ValueError("Project name cannot be empty after sanitization")
        
        # Create project path
        project_path = self.projects_root / sanitized_name
        
        # Check if project already exists
        if project_path.exists():
            # Check if directory has content other than conventions.json
            existing_files = list(project_path.iterdir())
            has_non_conventions_content = any(
                f.name != "conventions.json" for f in existing_files
            )
            
            if has_non_conventions_content:
                raise ValueError(f"Project '{sanitized_name}' already exists")
            
            # If only conventions.json exists, we can proceed (allows for convention-only initialization)
        
        # Create project structure
        create_project_structure(project_path)
        
        # Load existing conventions or create new ones
        existing_conventions = load_existing_conventions(project_path)
        conventions_summary = None
        
        if existing_conventions:
            conventions_summary = self._summarize_conventions(existing_conventions)
        
        # Create initial files
        create_initial_files(project_path)
        
        # Copy dataset if provided
        dataset_info = None
        if dataset_path:
            copied_dataset_path = copy_dataset_to_project(dataset_path, project_path)
            dataset_info = {
                "original_path": str(dataset_path),
                "project_path": str(copied_dataset_path.relative_to(project_path)),
                "filename": copied_dataset_path.name,
                "size_bytes": copied_dataset_path.stat().st_size
            }
        
        # Return project metadata
        return {
            "project_name": sanitized_name,
            "original_name": project_name,
            "project_path": str(project_path),
            "created": datetime.now().isoformat(),
            "dataset": dataset_info,
            "existing_conventions": existing_conventions is not None,
            "conventions_summary": conventions_summary,
            "directories": {
                "data": str(project_path / "data"),
                "history": str(project_path / "history"),
                "outputs": str(project_path / "outputs")
            },
            "files": {
                "conventions": str(project_path / "conventions.json"),
                "summary": str(project_path / "llm_summary.md"),
                "log": str(next(project_path.glob("llm_log_*.jsonl"), ""))
            }
        }
    
    def _summarize_conventions(self, conventions: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of existing conventions.
        
        Args:
            conventions: Conventions dictionary
            
        Returns:
            String summary of conventions
        """
        summary_parts = []
        
        if conventions.get("created"):
            summary_parts.append(f"Created: {conventions['created']}")
        
        if conventions.get("columns"):
            col_count = len(conventions["columns"])
            summary_parts.append(f"Column configurations: {col_count}")
        
        if conventions.get("notes"):
            summary_parts.append("Has project notes")
        
        if conventions.get("model_assumptions"):
            assumption_count = len(conventions["model_assumptions"])
            summary_parts.append(f"Model assumptions: {assumption_count}")
        
        if conventions.get("global_settings"):
            setting_count = len(conventions["global_settings"])
            summary_parts.append(f"Global settings: {setting_count}")
        
        return "; ".join(summary_parts) if summary_parts else "Empty conventions file"
    
    def list_projects(self) -> list[Dict[str, Any]]:
        """
        List all existing projects.
        
        Returns:
            List of project information dictionaries
        """
        projects = []
        
        if not self.projects_root.exists():
            return projects
        
        for project_dir in self.projects_root.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                project_info = {
                    "name": project_dir.name,
                    "path": str(project_dir),
                    "exists": True
                }
                
                # Check for conventions file
                conventions_path = project_dir / "conventions.json"
                if conventions_path.exists():
                    try:
                        with open(conventions_path, 'r', encoding='utf-8') as f:
                            conventions = json.load(f)
                            project_info["created"] = conventions.get("created")
                    except (json.JSONDecodeError, IOError):
                        pass
                
                # Check for data files
                data_dir = project_dir / "data"
                if data_dir.exists():
                    data_files = list(data_dir.glob("*"))
                    project_info["data_files"] = len(data_files)
                else:
                    project_info["data_files"] = 0
                
                projects.append(project_info)
        
        return sorted(projects, key=lambda x: x.get("created", ""))