"""
Tests for the updated PersistenceManager functionality.
"""

import json
import tempfile
import unittest
import shutil
from pathlib import Path
from datetime import datetime

from data_analysis_assistant.agent_core.persistence import PersistenceManager


class TestPersistenceManager(unittest.TestCase):
    """Test the PersistenceManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.projects_root = Path(self.temp_dir) / "projects"
        self.persistence = PersistenceManager(self.projects_root)
        
        # Create test dataset
        self.dataset_path = Path(self.temp_dir) / "test_data.csv"
        with open(self.dataset_path, 'w') as f:
            f.write("id,name,value\n1,test,100\n2,demo,200\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_project_basic(self):
        """Test basic project creation through PersistenceManager."""
        result = self.persistence.create_project("Test Project")
        
        self.assertEqual(result["project_name"], "test_project")
        self.assertEqual(result["original_name"], "Test Project")
        
        # Verify project exists
        project_path = Path(result["project_path"])
        self.assertTrue(project_path.exists())
        
        # Verify structure
        self.assertTrue((project_path / "data").exists())
        self.assertTrue((project_path / "history").exists())
        self.assertTrue((project_path / "outputs").exists())
        self.assertTrue((project_path / "conventions.json").exists())
    
    def test_create_project_with_dataset(self):
        """Test project creation with dataset."""
        result = self.persistence.create_project("Test Project", self.dataset_path)
        
        # Check dataset was copied
        self.assertIsNotNone(result["dataset"])
        project_path = Path(result["project_path"])
        dataset_copy = project_path / "data" / "test_data.csv"
        self.assertTrue(dataset_copy.exists())
    
    def test_list_projects(self):
        """Test project listing."""
        # Create a few projects
        self.persistence.create_project("Project One")
        self.persistence.create_project("Project Two", self.dataset_path)
        
        projects = self.persistence.list_projects()
        self.assertEqual(len(projects), 2)
        
        project_names = [p["name"] for p in projects]
        self.assertIn("project_one", project_names)
        self.assertIn("project_two", project_names)
    
    def test_save_and_load_script(self):
        """Test script saving and loading."""
        # Create project first
        self.persistence.create_project("Test Project")
        
        # Save a script
        test_code = "import pandas as pd\nprint('Hello World')"
        self.persistence.save_script("test_project", "analysis", test_code)
        
        # Load the script
        loaded_code = self.persistence.load_script("test_project", "analysis")
        self.assertEqual(loaded_code, test_code)
        
        # Verify file exists with .py extension
        project_path = self.projects_root / "test_project"
        script_path = project_path / "history" / "analysis.py"
        self.assertTrue(script_path.exists())
    
    def test_save_script_versioning(self):
        """Test script versioning when overwriting."""
        # Create project and save initial script
        self.persistence.create_project("Test Project")
        self.persistence.save_script("test_project", "analysis.py", "# Version 1")
        
        # Save updated script
        self.persistence.save_script("test_project", "analysis.py", "# Version 2")
        
        # Current script should have new content
        current_code = self.persistence.load_script("test_project", "analysis.py")
        self.assertEqual(current_code, "# Version 2")
        
        # Backup should exist
        project_path = self.projects_root / "test_project"
        history_dir = project_path / "history"
        backup_files = list(history_dir.glob("analysis_backup_*.py"))
        self.assertGreater(len(backup_files), 0)
    
    def test_log_llm_interaction(self):
        """Test LLM interaction logging."""
        # Create project
        self.persistence.create_project("Test Project")
        
        # Log an interaction
        interaction = {
            "role": "user",
            "content": "What is the mean of column A?",
            "response": "The mean of column A is 42.5"
        }
        self.persistence.log_llm_interaction("test_project", interaction)
        
        # Verify log file exists and has content
        project_path = self.projects_root / "test_project"
        log_files = list(project_path.glob("llm_log_*.jsonl"))
        self.assertGreater(len(log_files), 0)
        
        # Check log content
        with open(log_files[0], 'r') as f:
            logged_data = json.loads(f.read().strip())
        
        self.assertEqual(logged_data["role"], "user")
        self.assertEqual(logged_data["content"], "What is the mean of column A?")
        self.assertIn("timestamp", logged_data)
    
    def test_save_and_load_project_state(self):
        """Test project state persistence."""
        # Create project
        self.persistence.create_project("Test Project")
        
        # Save state
        test_state = {
            "current_dataset": "data.csv",
            "analysis_step": 3,
            "variables": {"mean_age": 42.5}
        }
        self.persistence.save_project_state("test_project", test_state)
        
        # Load state
        loaded_state = self.persistence.load_project_state("test_project")
        self.assertEqual(loaded_state, test_state)
    
    def test_load_nonexistent_project_state(self):
        """Test loading state for project without state file."""
        # Create project
        self.persistence.create_project("Test Project")
        
        # Load state (should return empty dict)
        state = self.persistence.load_project_state("test_project")
        self.assertEqual(state, {})
    
    def test_get_project_info(self):
        """Test comprehensive project information retrieval."""
        # Create project with dataset
        self.persistence.create_project("Test Project", self.dataset_path)
        
        # Add some content
        self.persistence.save_script("test_project", "analysis1.py", "# Script 1")
        self.persistence.save_script("test_project", "analysis2.py", "# Script 2")
        
        # Create output file
        project_path = self.projects_root / "test_project"
        output_file = project_path / "outputs" / "chart.png"
        output_file.touch()
        
        # Get project info
        info = self.persistence.get_project_info("test_project")
        
        self.assertEqual(info["name"], "test_project")
        self.assertTrue(info["exists"])
        self.assertTrue(info["has_conventions"])
        self.assertEqual(info["data_files"], 1)
        self.assertEqual(info["history_files"], 2)
        self.assertEqual(info["output_files"], 1)
        self.assertGreater(info["log_files"], 0)
        self.assertTrue(info["has_summary"])
    
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test operations on nonexistent project
        with self.assertRaises(ValueError):
            self.persistence.save_script("nonexistent", "test.py", "code")
        
        with self.assertRaises(FileNotFoundError):
            self.persistence.load_script("nonexistent", "test.py")
        
        with self.assertRaises(ValueError):
            self.persistence.log_llm_interaction("nonexistent", {})
        
        with self.assertRaises(ValueError):
            self.persistence.save_project_state("nonexistent", {})
        
        with self.assertRaises(FileNotFoundError):
            self.persistence.load_project_state("nonexistent")
        
        with self.assertRaises(FileNotFoundError):
            self.persistence.get_project_info("nonexistent")


if __name__ == "__main__":
    unittest.main()