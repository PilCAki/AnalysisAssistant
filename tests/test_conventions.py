"""
Tests for the ConventionManager functionality.
"""

import json
import tempfile
import unittest
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from data_analysis_assistant.agent_core.conventions import ConventionManager


class TestConventionManager(unittest.TestCase):
    """Test the ConventionManager class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.project_path.mkdir(parents=True)
        self.convention_manager = ConventionManager(self.project_path)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ConventionManager initialization."""
        self.assertEqual(self.convention_manager.project_path, self.project_path)
        self.assertIsInstance(self.convention_manager.conventions, dict)

        # Check default conventions structure
        expected_keys = [
            "time_column",
            "id_column",
            "target_column",
            "numeric_columns",
            "category_columns",
            "date_format",
            "null_handling",
            "index_column",
            "created",
            "last_updated",
            "inferred",
            "user_confirmed",
        ]
        for key in expected_keys:
            self.assertIn(key, self.convention_manager.conventions)

    def test_get_set_methods(self):
        """Test get and set methods."""
        # Test setting and getting values
        self.convention_manager.set("time_column", "date")
        self.assertEqual(self.convention_manager.get("time_column"), "date")

        # Test default value
        self.assertIsNone(self.convention_manager.get("nonexistent_key"))
        self.assertEqual(
            self.convention_manager.get("nonexistent_key", "default"), "default"
        )

    def test_save_and_load(self):
        """Test saving and loading conventions."""
        # Set some values
        self.convention_manager.set("time_column", "timestamp")
        self.convention_manager.set("id_column", "customer_id")

        # Save conventions
        self.convention_manager.save()

        # Verify file exists
        self.assertTrue(self.convention_manager.conventions_file.exists())

        # Create new manager and load
        new_manager = ConventionManager(self.project_path)
        self.assertEqual(new_manager.get("time_column"), "timestamp")
        self.assertEqual(new_manager.get("id_column"), "customer_id")

    def test_infer_time_column(self):
        """Test time column inference."""
        # Test with obvious datetime column name
        df1 = pd.DataFrame(
            {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "value": [1, 2, 3]}
        )
        df1["date"] = pd.to_datetime(df1["date"])

        time_col = self.convention_manager._infer_time_column(df1)
        self.assertEqual(time_col, "date")

        # Test with timestamp column name
        df2 = pd.DataFrame(
            {
                "timestamp": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
                "value": [1, 2],
            }
        )
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])

        time_col = self.convention_manager._infer_time_column(df2)
        self.assertEqual(time_col, "timestamp")

        # Test with no datetime column
        df3 = pd.DataFrame({"value": [1, 2, 3], "category": ["A", "B", "C"]})
        time_col = self.convention_manager._infer_time_column(df3)
        self.assertIsNone(time_col)

    def test_infer_id_column(self):
        """Test ID column inference."""
        # Test with obvious ID column name
        df1 = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "value": [1, 2, 3]})

        id_col = self.convention_manager._infer_id_column(df1)
        self.assertEqual(id_col, "customer_id")

        # Test with high cardinality column
        df2 = pd.DataFrame(
            {
                "unique_string": [f"item_{i}" for i in range(100)],
                "value": list(range(100)),
            }
        )

        id_col = self.convention_manager._infer_id_column(df2)
        self.assertEqual(id_col, "unique_string")

        # Test with no clear ID column
        df3 = pd.DataFrame({"value": [1, 2, 3], "category": ["A", "A", "B"]})
        id_col = self.convention_manager._infer_id_column(df3)
        self.assertIsNone(id_col)

    def test_classify_columns(self):
        """Test column classification."""
        df = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.0, 3.0],
                "category_col": ["A", "B", "A"],
                "low_cardinality_numeric": [
                    1,
                    1,
                    1,
                ],  # Should be treated as categorical (67% unique ratio, 2 unique vals)
                "string_col": ["text1", "text2", "text3"],
            }
        )

        numeric_cols, category_cols = self.convention_manager._classify_columns(df)

        self.assertIn("numeric_col", numeric_cols)
        self.assertIn("category_col", category_cols)
        self.assertIn("low_cardinality_numeric", category_cols)
        self.assertIn("string_col", category_cols)

    def test_infer_target_column(self):
        """Test target column inference."""
        # Test with obvious target column name
        df1 = pd.DataFrame({"features": [1, 2, 3], "churn": [0, 1, 0]})

        target_col = self.convention_manager._infer_target_column(df1)
        self.assertEqual(target_col, "churn")

        # Test with binary values
        df2 = pd.DataFrame(
            {"features": [1, 2, 3], "binary_outcome": ["Yes", "No", "Yes"]}
        )

        target_col = self.convention_manager._infer_target_column(df2)
        self.assertEqual(target_col, "binary_outcome")

        # Test with no clear target
        df3 = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        target_col = self.convention_manager._infer_target_column(df3)
        self.assertIsNone(target_col)

    def test_infer_date_format(self):
        """Test date format inference."""
        # Test YYYY-MM-DD format
        series1 = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        format1 = self.convention_manager._infer_date_format(series1)
        self.assertEqual(format1, "YYYY-MM-DD")

        # Test MM/DD/YYYY format
        series2 = pd.Series(["01/15/2023", "02/20/2023", "03/25/2023"])
        format2 = self.convention_manager._infer_date_format(series2)
        self.assertEqual(format2, "MM/DD/YYYY")

        # Test unrecognized format
        series3 = pd.Series(["January 1, 2023", "February 2, 2023"])
        format3 = self.convention_manager._infer_date_format(series3)
        self.assertIsNone(format3)

    def test_infer_from_data_integration(self):
        """Test full data inference integration."""
        # Create a comprehensive test dataset
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "signup_date": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "age": [25, 30, 35, 40, 45],
                "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
                "region": ["North", "South", "East", "West", "North"],
                "churn": [0, 1, 0, 1, 0],
            }
        )
        df["signup_date"] = pd.to_datetime(df["signup_date"])

        # Run inference
        inferred = self.convention_manager.infer_from_data(df)

        # Check inferred values
        self.assertEqual(inferred.get("time_column"), "signup_date")
        self.assertEqual(inferred.get("id_column"), "customer_id")
        self.assertEqual(inferred.get("target_column"), "churn")
        self.assertIn("age", inferred.get("numeric_columns", []))
        self.assertIn("income", inferred.get("numeric_columns", []))
        self.assertIn("region", inferred.get("category_columns", []))
        self.assertEqual(inferred.get("date_format"), "YYYY-MM-DD")

        # Check that conventions were updated
        self.assertEqual(self.convention_manager.get("time_column"), "signup_date")
        self.assertTrue(self.convention_manager.get("inferred"))

    def test_prompt_missing_conventions_console(self):
        """Test console-based prompting (mocked)."""
        # Set up a scenario with missing conventions
        self.convention_manager.set("time_column", "date")
        # Mark time_column as user confirmed to skip it in prompting
        if "user_confirmed" not in self.convention_manager.conventions:
            self.convention_manager.conventions["user_confirmed"] = {}
        self.convention_manager.conventions["user_confirmed"]["time_column"] = True

        # id_column and target_column are missing

        # Mock input responses
        responses = ["customer_id", "outcome"]
        response_iter = iter(responses)

        def mock_input(prompt):
            return next(response_iter)

        # Temporarily replace input function
        original_input = (
            __builtins__["input"]
            if isinstance(__builtins__, dict)
            else __builtins__.input
        )

        try:
            if isinstance(__builtins__, dict):
                __builtins__["input"] = mock_input
            else:
                __builtins__.input = mock_input

            # Capture print output (simple approach)
            user_responses = self.convention_manager.prompt_missing_conventions()

            # Check responses were processed
            self.assertEqual(user_responses.get("id_column"), "customer_id")
            self.assertEqual(user_responses.get("target_column"), "outcome")

            # Check conventions were updated
            self.assertEqual(self.convention_manager.get("id_column"), "customer_id")
            self.assertEqual(self.convention_manager.get("target_column"), "outcome")

        finally:
            # Restore original input function
            if isinstance(__builtins__, dict):
                __builtins__["input"] = original_input
            else:
                __builtins__.input = original_input

    def test_prompt_missing_conventions_ui_callback(self):
        """Test UI callback-based prompting."""

        # Set up callback function
        def mock_ui_callback(prompt, description, current_value):
            if "time" in prompt.lower():
                return "timestamp"
            elif "id" in prompt.lower():
                return "user_id"
            elif "target" in prompt.lower():
                return "label"
            return ""

        user_responses = self.convention_manager.prompt_missing_conventions(
            mock_ui_callback
        )

        # Check responses
        self.assertEqual(user_responses.get("time_column"), "timestamp")
        self.assertEqual(user_responses.get("id_column"), "user_id")
        self.assertEqual(user_responses.get("target_column"), "label")

    def test_user_confirmed_tracking(self):
        """Test tracking of user-confirmed conventions."""
        # Initially no confirmations
        self.assertEqual(self.convention_manager.get("user_confirmed", {}), {})

        # Use UI callback to confirm some values
        def mock_ui_callback(prompt, description, current_value):
            return "confirmed_value"

        self.convention_manager.prompt_missing_conventions(mock_ui_callback)

        # Check that confirmations were tracked
        confirmed = self.convention_manager.get("user_confirmed", {})
        self.assertTrue(confirmed.get("time_column"))
        self.assertTrue(confirmed.get("id_column"))
        self.assertTrue(confirmed.get("target_column"))

    def test_load_invalid_conventions_file(self):
        """Test handling of invalid conventions.json file."""
        # Create invalid JSON file
        with open(self.convention_manager.conventions_file, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully and use defaults
        new_manager = ConventionManager(self.project_path)
        self.assertIsNotNone(new_manager.conventions)
        self.assertIn("time_column", new_manager.conventions)


if __name__ == "__main__":
    unittest.main()
