"""
Conventions manager - Convention inference, user prompts, and project config.

This module handles project-specific conventions and configuration,
learning user preferences and maintaining consistency across sessions.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import pandas as pd
import numpy as np


class ConventionManager:
    """
    Manages project conventions and user preferences.
    
    Handles automatic convention inference from data, user prompting for
    missing conventions, and persistence of conventions to JSON files.
    """
    
    def __init__(self, project_path: Union[str, Path]):
        """
        Initialize the Convention Manager.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        self.conventions_file = self.project_path / "conventions.json"
        self.conventions = self._get_default_conventions()
        
        # Load existing conventions if available
        if self.conventions_file.exists():
            self.load()
    
    def _get_default_conventions(self) -> Dict[str, Any]:
        """Get default convention schema."""
        return {
            "time_column": None,
            "id_column": None,
            "target_column": None,
            "numeric_columns": [],
            "category_columns": [],
            "date_format": "YYYY-MM-DD",
            "null_handling": "drop",
            "index_column": None,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inferred": False,
            "user_confirmed": {}
        }
    
    def load(self) -> None:
        """Load existing conventions.json if available."""
        if self.conventions_file.exists():
            try:
                with open(self.conventions_file, 'r', encoding='utf-8') as f:
                    loaded_conventions = json.load(f)
                    # Merge with defaults to ensure all required keys exist
                    self.conventions.update(loaded_conventions)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load conventions file: {e}")
    
    def save(self) -> None:
        """Write conventions to conventions.json."""
        self.conventions["last_updated"] = datetime.now().isoformat()
        
        # Ensure project directory exists
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        with open(self.conventions_file, 'w', encoding='utf-8') as f:
            json.dump(self.conventions, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a convention value."""
        return self.conventions.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a convention value."""
        self.conventions[key] = value
        self.conventions["last_updated"] = datetime.now().isoformat()
    
    def infer_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Guess conventions based on data profile using heuristics.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary of inferred conventions
        """
        inferred = {}
        
        # Infer time column
        time_col = self._infer_time_column(df)
        if time_col:
            inferred["time_column"] = time_col
        
        # Infer ID column
        id_col = self._infer_id_column(df)
        if id_col:
            inferred["id_column"] = id_col
        
        # Classify columns as numeric or categorical
        numeric_cols, category_cols = self._classify_columns(df)
        inferred["numeric_columns"] = numeric_cols
        inferred["category_columns"] = category_cols
        
        # Attempt to infer target column (binary classification target)
        target_col = self._infer_target_column(df)
        if target_col:
            inferred["target_column"] = target_col
        
        # Infer date format if time column found
        if time_col and time_col in df.columns:
            date_format = self._infer_date_format(df[time_col])
            if date_format:
                inferred["date_format"] = date_format
        
        # Update conventions with inferred values
        for key, value in inferred.items():
            if value is not None:
                self.conventions[key] = value
        
        self.conventions["inferred"] = True
        self.conventions["last_updated"] = datetime.now().isoformat()
        
        return inferred
    
    def _infer_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Infer time/date column from DataFrame."""
        time_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'day', 'month', 'year']
        
        # Check column names first
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in time_keywords):
                # Verify the column actually contains date-like data
                if self._is_datetime_like(df[col]):
                    return col
        
        # Check columns for datetime-like content
        for col in df.columns:
            if self._is_datetime_like(df[col]):
                return col
        
        return None
    
    def _infer_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Infer ID column from DataFrame."""
        id_keywords = ['id', 'identifier', 'key', 'customer', 'user', 'account']
        
        # Check for ID-like column names
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in id_keywords):
                return col
        
        # Look for high-cardinality columns that might be IDs
        for col in df.columns:
            if len(df) > 0:
                unique_ratio = df[col].nunique() / len(df)
                # High cardinality (>80% unique) and not obviously numeric
                if unique_ratio > 0.8 and not pd.api.types.is_numeric_dtype(df[col]):
                    return col
        
        return None
    
    def _classify_columns(self, df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Classify columns as numeric or categorical."""
        numeric_cols = []
        category_cols = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (low unique count)
                if len(df) > 0:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5 and df[col].nunique() < 10:
                        category_cols.append(col)
                    else:
                        numeric_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                category_cols.append(col)
        
        return numeric_cols, category_cols
    
    def _infer_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Attempt to infer target column (binary classification)."""
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'churn', 'conversion']
        
        # Check for target-like column names
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in target_keywords):
                return col
        
        # Look for binary columns
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                # Check if it looks like a binary target (0/1, True/False, Yes/No, etc.)
                vals_set = set(str(v).lower() for v in unique_vals)
                binary_patterns = [
                    {'0', '1'}, {'true', 'false'}, {'yes', 'no'}, 
                    {'y', 'n'}, {'positive', 'negative'}, {'pos', 'neg'}
                ]
                if any(vals_set == pattern for pattern in binary_patterns):
                    return col
        
        return None
    
    def _infer_date_format(self, series: pd.Series) -> Optional[str]:
        """Infer date format from a pandas Series."""
        # Try to detect common date formats
        sample_values = series.dropna().head(10).astype(str)
        
        format_patterns = [
            (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
            (r'^\d{2}/\d{2}/\d{4}$', 'MM/DD/YYYY'),
            (r'^\d{4}/\d{2}/\d{2}$', 'YYYY/MM/DD'),
            (r'^\d{2}-\d{2}-\d{4}$', 'MM-DD-YYYY'),
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', 'YYYY-MM-DD HH:MM:SS'),
        ]
        
        for pattern, format_str in format_patterns:
            if all(re.match(pattern, str(val)) for val in sample_values if str(val) != 'nan'):
                return format_str
        
        return None
    
    def _is_datetime_like(self, series: pd.Series) -> bool:
        """Check if a series contains datetime-like data."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(series):
            return False
        
        # Try to parse a sample as datetime
        sample = series.dropna().head(5)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def prompt_missing_conventions(
        self, 
        ui_callback: Optional[Callable[[str, str, Optional[str]], str]] = None
    ) -> Dict[str, str]:
        """
        Prompt user to confirm or fill in missing conventions.
        
        Args:
            ui_callback: Optional callback function for UI prompts.
                         Should accept (prompt, description, current_value) and return user response.
                         If None, falls back to input() prompts.
        
        Returns:
            Dictionary of user responses
        """
        responses = {}
        
        # Check each key convention
        convention_prompts = {
            "time_column": {
                "prompt": "What is your time/date column name?",
                "description": "Column containing timestamps or dates for time series analysis"
            },
            "id_column": {
                "prompt": "What is your ID column name?", 
                "description": "Column containing unique identifiers for each record"
            },
            "target_column": {
                "prompt": "What is your target column name?",
                "description": "Column containing the outcome/target variable for prediction"
            }
        }
        
        for key, prompt_info in convention_prompts.items():
            current_value = self.conventions.get(key)
            
            # Skip if already confirmed by user
            if self.conventions.get("user_confirmed", {}).get(key):
                continue
            
            prompt_text = prompt_info["prompt"]
            description = prompt_info["description"]
            
            if current_value:
                prompt_text += f" (detected: {current_value})"
            
            if ui_callback:
                response = ui_callback(prompt_text, description, current_value)
            else:
                # Fallback to console input
                print(f"\n{description}")
                response = input(f"{prompt_text}: ").strip()
            
            if response:
                responses[key] = response
                self.set(key, response)
                # Mark as user confirmed
                if "user_confirmed" not in self.conventions:
                    self.conventions["user_confirmed"] = {}
                self.conventions["user_confirmed"][key] = True
        
        return responses