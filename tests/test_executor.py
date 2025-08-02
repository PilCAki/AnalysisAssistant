"""
Tests for the code execution engine.

Tests cover all aspects of the code executor including successful execution,
error handling, output capture, globals injection, and history saving.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_analysis_assistant.agent_core.executor import (
    CodeExecutor, 
    ExecutionResult, 
    run_code
)


class TestExecutionResult(unittest.TestCase):
    """Test the ExecutionResult model."""
    
    def test_execution_result_creation(self):
        """Test creating ExecutionResult with all fields."""
        result = ExecutionResult(
            success=True,
            stdout="Hello, World!",
            stderr="",
            exception=None,
            result=42,
            code="print('Hello, World!')",
            execution_time=0.1
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Hello, World!")
        self.assertEqual(result.stderr, "")
        self.assertIsNone(result.exception)
        self.assertEqual(result.result, 42)
        self.assertEqual(result.code, "print('Hello, World!')")
        self.assertEqual(result.execution_time, 0.1)
    
    def test_execution_result_defaults(self):
        """Test ExecutionResult with minimal required fields."""
        result = ExecutionResult(
            success=False,
            code="invalid code",
            execution_time=0.05
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertIsNone(result.exception)
        self.assertIsNone(result.result)


class TestRunCode(unittest.TestCase):
    """Test the run_code function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for history
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_successful_code_execution(self):
        """Test successful code execution with stdout capture."""
        code = """
print("Hello, World!")
print("Line 2")
x = 42
        """
        
        result = run_code(code)
        
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Hello, World!\nLine 2\n")
        self.assertEqual(result.stderr, "")
        self.assertIsNone(result.exception)
        self.assertEqual(result.code, code)
        self.assertGreater(result.execution_time, 0)
    
    def test_code_execution_with_stderr(self):
        """Test code execution that writes to stderr."""
        code = """
import sys
print("stdout message")
sys.stderr.write("stderr message\\n")
        """
        
        result = run_code(code)
        
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "stdout message\n")
        self.assertEqual(result.stderr, "stderr message\n")
        self.assertIsNone(result.exception)
    
    def test_code_execution_with_exception(self):
        """Test code execution that raises an exception."""
        code = """
x = 1 / 0  # This will raise ZeroDivisionError
        """
        
        result = run_code(code)
        
        self.assertFalse(result.success)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertIsNotNone(result.exception)
        
        # Check exception details
        exception = result.exception
        self.assertEqual(exception["type"], "ZeroDivisionError")
        self.assertIn("division by zero", exception["message"])
        self.assertIn("ZeroDivisionError:", exception["summary"])
        self.assertIn("Traceback", exception["traceback"])
        self.assertIn("ZeroDivisionError", exception["traceback"])
    
    def test_code_execution_with_syntax_error(self):
        """Test code execution with syntax error."""
        code = """
if True
    print("Missing colon")
        """
        
        result = run_code(code)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)
        self.assertEqual(result.exception["type"], "SyntaxError")
        self.assertIn("SyntaxError:", result.exception["summary"])
    
    def test_globals_injection(self):
        """Test injecting global variables into execution context."""
        code = """
result = x + y
print(f"Result: {result}")
        """
        
        globals_dict = {"x": 10, "y": 20}
        result = run_code(code, globals_dict)
        
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Result: 30\n")
        self.assertIsNone(result.exception)
    
    def test_globals_with_dataframe(self):
        """Test injecting a pandas DataFrame (common use case)."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")
        
        code = """
print(f"DataFrame shape: {df.shape}")
df['new_column'] = df['a'] * 2
print(f"New column created")
        """
        
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        globals_dict = {"df": df}
        
        result = run_code(code, globals_dict)
        
        self.assertTrue(result.success)
        self.assertIn("DataFrame shape: (3, 2)", result.stdout)
        self.assertIn("New column created", result.stdout)
    
    def test_execution_timing(self):
        """Test that execution time is measured."""
        code = """
import time
time.sleep(0.01)  # Sleep for 10ms
        """
        
        result = run_code(code)
        
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0.009)  # Should be at least 10ms
    
    def test_history_saving(self):
        """Test that successful code is saved to history."""
        code = """
print("This code should be saved to history")
x = 42
        """
        
        result = run_code(code)
        
        self.assertTrue(result.success)
        
        # Check if history directory was created
        history_dir = Path("history")
        self.assertTrue(history_dir.exists())
        
        # Check if a file was created
        history_files = list(history_dir.glob("code_*.py"))
        self.assertGreater(len(history_files), 0)
        
        # Check file content
        latest_file = max(history_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            content = f.read()
        
        self.assertIn(code.strip(), content)
        self.assertIn("# Code executed at", content)
    
    def test_history_not_saved_on_error(self):
        """Test that failed code is not saved to history."""
        # Clear any existing history
        history_dir = Path("history")
        if history_dir.exists():
            shutil.rmtree(history_dir)
        
        code = """
raise ValueError("This should not be saved")
        """
        
        result = run_code(code)
        
        self.assertFalse(result.success)
        
        # Check that no history was saved
        if history_dir.exists():
            history_files = list(history_dir.glob("code_*.py"))
            self.assertEqual(len(history_files), 0)


class TestCodeExecutor(unittest.TestCase):
    """Test the CodeExecutor class for backward compatibility."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        self.executor = CodeExecutor()
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_executor_initialization(self):
        """Test CodeExecutor initialization."""
        executor = CodeExecutor()
        self.assertEqual(executor.execution_context, {})
        self.assertEqual(executor.execution_history, [])
    
    def test_executor_code_execution(self):
        """Test code execution through CodeExecutor."""
        code = """
print("Hello from executor!")
result_var = 100
        """
        
        result = self.executor.execute_code(code)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Hello from executor!\n")
        self.assertEqual(len(self.executor.execution_history), 1)
    
    def test_executor_with_context(self):
        """Test executor with context variables."""
        code = """
output = input_value * 2
print(f"Doubled: {output}")
        """
        
        context = {"input_value": 21}
        result = self.executor.execute_code(code, context)
        
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Doubled: 42\n")
    
    def test_executor_persistent_context(self):
        """Test that executor maintains persistent context."""
        # First execution sets a variable
        code1 = "persistent_var = 'Hello'"
        result1 = self.executor.execute_code(code1)
        self.assertTrue(result1.success)
        
        # Second execution should have access to the variable
        code2 = "print(f'{persistent_var}, World!')"
        result2 = self.executor.execute_code(code2)
        self.assertTrue(result2.success)
        self.assertEqual(result2.stdout, "Hello, World!\n")


if __name__ == "__main__":
    unittest.main()