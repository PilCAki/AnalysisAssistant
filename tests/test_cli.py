"""
Tests for the CLI entry point functionality.

Tests that the CLI supports the --launch flag as requested in the issue.
"""

import unittest
import subprocess
import sys
from pathlib import Path


class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""
    
    def setUp(self):
        """Set up test cases."""
        self.repo_root = Path(__file__).parent.parent
        self.cli_module = "data_analysis_assistant.cli"
    
    def test_cli_help(self):
        """Test that CLI help shows --launch option."""
        result = subprocess.run(
            [sys.executable, "-m", self.cli_module, "--help"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--launch", result.stdout)
        self.assertIn("Launch the Streamlit UI", result.stdout)
    
    def test_cli_unknown_args(self):
        """Test that unknown arguments show error."""
        result = subprocess.run(
            [sys.executable, "-m", self.cli_module, "--unknown-arg"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)
    
    def test_cli_launch_without_streamlit(self):
        """Test that --launch shows helpful error when Streamlit not available."""
        result = subprocess.run(
            [sys.executable, "-m", self.cli_module, "--launch"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        # Should exit with error code 1 when Streamlit not installed
        self.assertEqual(result.returncode, 1)
        self.assertIn("Streamlit is not installed", result.stdout)
        self.assertIn("pip install streamlit", result.stdout)
    
    def test_backward_compatibility_subcommand(self):
        """Test that the old 'launch' subcommand still works."""
        result = subprocess.run(
            [sys.executable, "-m", self.cli_module, "launch"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        # Should exit with error code 1 when Streamlit not installed (same as --launch)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Streamlit is not installed", result.stdout)


if __name__ == "__main__":
    unittest.main()