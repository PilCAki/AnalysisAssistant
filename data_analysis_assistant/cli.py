"""
Command-line interface for the AnalysisAssistant.

This module provides CLI commands for launching the application
and managing projects from the command line.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def launch_streamlit():
    """Launch the Streamlit application."""
    # Check if Streamlit is available (try import first, then shutil.which)
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed. Run `pip install streamlit`.")
        sys.exit(1)
    
    try:
        # Get the path to the Streamlit app
        package_dir = Path(__file__).parent
        app_path = package_dir / "ui" / "app_streamlit.py"
        
        # Launch Streamlit
        cmd = ["streamlit", "run", str(app_path)]
        subprocess.run(cmd, check=True)
        
    except FileNotFoundError:
        print("❌ Streamlit command not found. Run `pip install streamlit`.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)


def create_project(project_name: str, dataset_path: str = None):
    """
    Create a new analysis project.
    
    Args:
        project_name: Name of the project to create
        dataset_path: Optional path to initial dataset file
    """
    from .agent_core.persistence import PersistenceManager
    
    try:
        persistence = PersistenceManager()
        
        print(f"Creating project: {project_name}")
        
        # Initialize the project
        result = persistence.create_project(project_name, dataset_path)
        
        print(f"✅ Project '{result['project_name']}' created successfully!")
        print(f"   Location: {result['project_path']}")
        
        if result.get("dataset"):
            print(f"   Dataset: {result['dataset']['filename']} ({result['dataset']['size_bytes']} bytes)")
        
        if result.get("existing_conventions"):
            print(f"   Found existing conventions: {result['conventions_summary']}")
        
        print("\nProject structure:")
        for dir_name, dir_path in result["directories"].items():
            print(f"   📁 {dir_name}/")
        
        print("\nInitial files:")
        for file_name, file_path in result["files"].items():
            if file_path:  # Only show files that were created
                print(f"   📄 {Path(file_path).name}")
        
    except ValueError as e:
        print(f"❌ Error creating project: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"❌ Dataset file error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


def list_projects():
    """
    List all available projects.
    """
    from .agent_core.persistence import PersistenceManager
    
    try:
        persistence = PersistenceManager()
        projects = persistence.list_projects()
        
        if not projects:
            print("No projects found.")
            print("Create a new project with: data-assist project create <project_name>")
            return 0
        
        print(f"Found {len(projects)} project(s):")
        print()
        
        for project in projects:
            print(f"📁 {project['name']}")
            if project.get('created'):
                print(f"   Created: {project['created']}")
            print(f"   Path: {project['path']}")
            print(f"   Data files: {project.get('data_files', 0)}")
            print()
        
    except Exception as e:
        print(f"❌ Error listing projects: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AnalysisAssistant - AI-powered data analysis tool"
    )
    
    # Add the --launch flag as requested in the issue
    parser.add_argument(
        "--launch", action="store_true", help="Launch the Streamlit UI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Launch command (keep existing subcommand for backward compatibility)
    launch_parser = subparsers.add_parser("launch", help="Launch the Streamlit UI")
    
    # Project management commands
    project_parser = subparsers.add_parser("project", help="Project management")
    project_subparsers = project_parser.add_subparsers(dest="project_action")
    
    create_parser = project_subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    create_parser.add_argument(
        "--dataset", "-d", 
        help="Path to initial dataset file (CSV, XLSX, etc.)"
    )
    
    list_parser = project_subparsers.add_parser("list", help="List all projects")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for --launch flag first (as requested in the issue)
    if args.launch:
        launch_streamlit()
    elif args.command == "launch":
        launch_streamlit()
    elif args.command == "project":
        if args.project_action == "create":
            exit_code = create_project(args.name, args.dataset)
            sys.exit(exit_code)
        elif args.project_action == "list":
            exit_code = list_projects()
            sys.exit(exit_code)
        else:
            project_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()