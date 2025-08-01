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


def create_project(project_name: str):
    """
    Create a new analysis project.
    
    TODO: Implement project creation logic
    """
    print(f"Creating project: {project_name}")
    # TODO: Use PersistenceManager to create project structure
    pass


def list_projects():
    """
    List all available projects.
    
    TODO: Implement project listing
    """
    print("Available projects:")
    # TODO: Use PersistenceManager to list projects
    pass


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
            create_project(args.name)
        elif args.project_action == "list":
            list_projects()
        else:
            project_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()