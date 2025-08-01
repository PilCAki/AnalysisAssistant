"""
Tests for project initialization functionality.
"""

import json
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_analysis_assistant.agent_core.project_init import (
    sanitize_project_name,
    create_conventions_scaffold,
    create_project_structure,
    create_initial_files,
    copy_dataset_to_project,
    load_existing_conventions,
    ProjectInitializer
)


class TestProjectNameSanitization(unittest.TestCase):
    """Test project name sanitization functionality."""
    
    def test_basic_sanitization(self):
        """Test basic name sanitization."""
        self.assertEqual(sanitize_project_name("My Project"), "my_project")
        self.assertEqual(sanitize_project_name("  Test  Name  "), "test_name")
        self.assertEqual(sanitize_project_name("Project123"), "project123")
    
    def test_special_characters(self):
        """Test removal of special characters."""
        self.assertEqual(sanitize_project_name("Project@#$%Name"), "projectname")
        self.assertEqual(sanitize_project_name("My-Project_Name"), "my-project_name")
        self.assertEqual(sanitize_project_name("file.name.ext"), "file.name.ext")
    
    def test_edge_cases(self):
        """Test edge cases in name sanitization."""
        with self.assertRaises(ValueError):
            sanitize_project_name("")
        with self.assertRaises(ValueError):
            sanitize_project_name("   ")
        with self.assertRaises(ValueError):
            sanitize_project_name("@#$%")
        
        # Test long names
        long_name = "a" * 150
        result = sanitize_project_name(long_name)
        self.assertLessEqual(len(result), 100)
        self.assertEqual(result, "a" * 100)
    
    def test_leading_trailing_cleanup(self):
        """Test cleanup of leading/trailing special characters."""
        self.assertEqual(sanitize_project_name("_test_"), "test")
        self.assertEqual(sanitize_project_name(".test."), "test")
        self.assertEqual(sanitize_project_name("__test__"), "test")


class TestConventionsScaffold(unittest.TestCase):
    """Test conventions scaffolding functionality."""
    
    def test_scaffold_structure(self):
        """Test that scaffold has required structure."""
        scaffold = create_conventions_scaffold()
        
        required_keys = ["created", "columns", "notes", "model_assumptions", "global_settings"]
        for key in required_keys:
            self.assertIn(key, scaffold)
    
    def test_scaffold_types(self):
        """Test that scaffold has correct data types."""
        scaffold = create_conventions_scaffold()
        
        self.assertIsInstance(scaffold["created"], str)
        self.assertIsInstance(scaffold["columns"], dict)
        self.assertIsInstance(scaffold["notes"], str)
        self.assertIsInstance(scaffold["model_assumptions"], dict)
        self.assertIsInstance(scaffold["global_settings"], dict)
    
    def test_created_timestamp_format(self):
        """Test that created timestamp is in ISO format."""
        scaffold = create_conventions_scaffold()
        created = scaffold["created"]
        
        # Should be able to parse as ISO format
        from datetime import datetime
        parsed = datetime.fromisoformat(created)
        self.assertIsInstance(parsed, datetime)


class TestProjectStructureCreation(unittest.TestCase):
    """Test project directory structure creation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_project_structure(self):
        """Test creation of project directory structure."""
        create_project_structure(self.project_path)
        
        # Check that project directory exists
        self.assertTrue(self.project_path.exists())
        self.assertTrue(self.project_path.is_dir())
        
        # Check that subdirectories exist
        expected_dirs = ["data", "history", "outputs"]
        for dirname in expected_dirs:
            dir_path = self.project_path / dirname
            self.assertTrue(dir_path.exists(), f"Directory {dirname} should exist")
            self.assertTrue(dir_path.is_dir(), f"{dirname} should be a directory")
    
    def test_create_initial_files(self):
        """Test creation of initial project files."""
        # First create the structure
        create_project_structure(self.project_path)
        
        # Then create initial files
        create_initial_files(self.project_path)
        
        # Check that files exist
        conventions_path = self.project_path / "conventions.json"
        summary_path = self.project_path / "llm_summary.md"
        
        self.assertTrue(conventions_path.exists())
        self.assertTrue(summary_path.exists())
        
        # Check that at least one log file exists
        log_files = list(self.project_path.glob("llm_log_*.jsonl"))
        self.assertGreater(len(log_files), 0)
        
        # Check conventions.json content
        with open(conventions_path, 'r') as f:
            conventions = json.load(f)
        
        required_keys = ["created", "columns", "notes", "model_assumptions", "global_settings"]
        for key in required_keys:
            self.assertIn(key, conventions)


class TestDatasetCopying(unittest.TestCase):
    """Test dataset copying functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.project_path.mkdir(parents=True)
        (self.project_path / "data").mkdir()
        
        # Create test dataset file
        self.dataset_path = Path(self.temp_dir) / "test_dataset.csv"
        with open(self.dataset_path, 'w') as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_copy_dataset_success(self):
        """Test successful dataset copying."""
        result_path = copy_dataset_to_project(self.dataset_path, self.project_path)
        
        # Check that file was copied to data directory
        expected_path = self.project_path / "data" / "test_dataset.csv"
        self.assertEqual(result_path, expected_path)
        self.assertTrue(result_path.exists())
        
        # Check that content is identical
        with open(self.dataset_path, 'r') as original:
            with open(result_path, 'r') as copied:
                self.assertEqual(original.read(), copied.read())
    
    def test_copy_dataset_file_not_found(self):
        """Test error when dataset file doesn't exist."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.csv"
        
        with self.assertRaises(FileNotFoundError):
            copy_dataset_to_project(nonexistent_path, self.project_path)
    
    def test_copy_dataset_unsupported_format(self):
        """Test error for unsupported file formats."""
        unsupported_path = Path(self.temp_dir) / "test.txt"
        unsupported_path.touch()
        
        with self.assertRaises(ValueError) as cm:
            copy_dataset_to_project(unsupported_path, self.project_path)
        
        self.assertIn("Unsupported dataset format", str(cm.exception))
    
    def test_copy_dataset_name_collision(self):
        """Test handling of filename collisions."""
        # Copy once
        first_copy = copy_dataset_to_project(self.dataset_path, self.project_path)
        
        # Copy again - should get versioned name
        second_copy = copy_dataset_to_project(self.dataset_path, self.project_path)
        
        self.assertNotEqual(first_copy, second_copy)
        self.assertTrue(first_copy.exists())
        self.assertTrue(second_copy.exists())
        self.assertEqual(second_copy.name, "test_dataset_1.csv")


class TestConventionsLoading(unittest.TestCase):
    """Test loading of existing conventions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.project_path.mkdir()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_existing_conventions(self):
        """Test loading valid existing conventions."""
        conventions_data = {
            "created": "2024-01-01T12:00:00",
            "columns": {"col1": "numeric"},
            "notes": "Test notes",
            "model_assumptions": {"assumption1": "value1"},
            "global_settings": {"setting1": "value1"}
        }
        
        conventions_path = self.project_path / "conventions.json"
        with open(conventions_path, 'w') as f:
            json.dump(conventions_data, f)
        
        result = load_existing_conventions(self.project_path)
        self.assertEqual(result, conventions_data)
    
    def test_load_nonexistent_conventions(self):
        """Test loading when conventions file doesn't exist."""
        result = load_existing_conventions(self.project_path)
        self.assertIsNone(result)
    
    def test_load_invalid_conventions(self):
        """Test loading invalid JSON conventions file."""
        conventions_path = self.project_path / "conventions.json"
        with open(conventions_path, 'w') as f:
            f.write("invalid json content")
        
        # Should return None for invalid JSON
        result = load_existing_conventions(self.project_path)
        self.assertIsNone(result)


class TestProjectInitializer(unittest.TestCase):
    """Test the main ProjectInitializer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.projects_root = Path(self.temp_dir) / "projects"
        self.initializer = ProjectInitializer(self.projects_root)
        
        # Create test dataset
        self.dataset_path = Path(self.temp_dir) / "test_data.csv"
        with open(self.dataset_path, 'w') as f:
            f.write("name,age,city\nJohn,25,NYC\nJane,30,LA\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialize_project_basic(self):
        """Test basic project initialization."""
        result = self.initializer.initialize_project("Test Project")
        
        # Check returned metadata
        self.assertEqual(result["project_name"], "test_project")
        self.assertEqual(result["original_name"], "Test Project")
        self.assertIn("created", result)
        self.assertFalse(result["existing_conventions"])
        
        # Check that project directory was created
        project_path = Path(result["project_path"])
        self.assertTrue(project_path.exists())
        
        # Check directory structure
        self.assertTrue((project_path / "data").exists())
        self.assertTrue((project_path / "history").exists())
        self.assertTrue((project_path / "outputs").exists())
        
        # Check files
        self.assertTrue((project_path / "conventions.json").exists())
        self.assertTrue((project_path / "llm_summary.md").exists())
        
        # Check that at least one log file exists
        log_files = list(project_path.glob("llm_log_*.jsonl"))
        self.assertGreater(len(log_files), 0)
    
    def test_initialize_project_with_dataset(self):
        """Test project initialization with dataset."""
        result = self.initializer.initialize_project("Test Project", self.dataset_path)
        
        # Check dataset info in result
        self.assertIsNotNone(result["dataset"])
        self.assertEqual(result["dataset"]["filename"], "test_data.csv")
        self.assertGreater(result["dataset"]["size_bytes"], 0)
        
        # Check that dataset was copied
        project_path = Path(result["project_path"])
        dataset_copy = project_path / "data" / "test_data.csv"
        self.assertTrue(dataset_copy.exists())
        
        # Check content
        with open(dataset_copy, 'r') as f:
            content = f.read()
        self.assertIn("name,age,city", content)
    
    def test_initialize_project_already_exists(self):
        """Test error when project already exists."""
        # Create project first time
        self.initializer.initialize_project("Test Project")
        
        # Try to create again - should raise error
        with self.assertRaises(ValueError) as cm:
            self.initializer.initialize_project("Test Project")
        
        self.assertIn("already exists", str(cm.exception))
    
    def test_initialize_project_empty_name(self):
        """Test error with empty project name."""
        with self.assertRaises(ValueError) as cm:
            self.initializer.initialize_project("")
        
        self.assertIn("cannot be empty", str(cm.exception))
        
        # Also test with whitespace-only name
        with self.assertRaises(ValueError) as cm:
            self.initializer.initialize_project("   ")
        
        self.assertIn("cannot be empty", str(cm.exception))
    
    def test_list_projects_empty(self):
        """Test listing projects when none exist."""
        projects = self.initializer.list_projects()
        self.assertEqual(projects, [])
    
    def test_list_projects_with_projects(self):
        """Test listing existing projects."""
        # Create a couple of projects
        self.initializer.initialize_project("Project One")
        self.initializer.initialize_project("Project Two", self.dataset_path)
        
        projects = self.initializer.list_projects()
        
        self.assertEqual(len(projects), 2)
        
        # Check project names
        project_names = [p["name"] for p in projects]
        self.assertIn("project_one", project_names)
        self.assertIn("project_two", project_names)
        
        # Check data file counts
        for project in projects:
            if project["name"] == "project_one":
                self.assertEqual(project["data_files"], 0)
            elif project["name"] == "project_two":
                self.assertEqual(project["data_files"], 1)
    
    def test_conventions_summary(self):
        """Test conventions summary generation."""
        # Create a project with a different name to avoid conflicts
        project_name = "conventions_test_project"
        project_path = self.projects_root / project_name
        project_path.mkdir(parents=True)
        
        conventions_data = {
            "created": "2024-01-01T12:00:00",
            "columns": {"col1": "numeric", "col2": "text"},
            "notes": "Some notes",
            "model_assumptions": {"assumption1": "value1"},
            "global_settings": {"setting1": "value1", "setting2": "value2"}
        }
        
        conventions_path = project_path / "conventions.json"
        with open(conventions_path, 'w') as f:
            json.dump(conventions_data, f)
        
        # Initialize project (should detect existing conventions)
        result = self.initializer.initialize_project(project_name)
        
        self.assertTrue(result["existing_conventions"])
        self.assertIsNotNone(result["conventions_summary"])
        
        summary = result["conventions_summary"]
        self.assertIn("Created: 2024-01-01T12:00:00", summary)
        self.assertIn("Column configurations: 2", summary)
        self.assertIn("Has project notes", summary)
        self.assertIn("Model assumptions: 1", summary)
        self.assertIn("Global settings: 2", summary)


if __name__ == "__main__":
    unittest.main()