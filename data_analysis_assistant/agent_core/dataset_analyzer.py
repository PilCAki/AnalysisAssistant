"""
Dataset analysis and metadata generation module.

This module provides comprehensive dataset analysis functionality,
including data profiling, type inference, and metadata generation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class DatasetAnalyzer:
    """
    Analyzes datasets and generates comprehensive metadata for LLM prompts.
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_dataset(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a dataset.
        
        Args:
            df: Pandas DataFrame to analyze
            filename: Original filename of the dataset
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        analysis = {
            "filename": filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "column_summary": self._analyze_columns(df),
            "overall_stats": self._get_overall_stats(df),
            "data_quality": self._assess_data_quality(df),
            "preview_data": self._get_preview_data(df)
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column in detail."""
        column_analysis = []
        
        for col in df.columns:
            col_data = df[col]
            col_info = {
                "name": col,
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.count()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(col_data.isnull().sum() / len(df) * 100),
                "unique_count": int(col_data.nunique()),
                "unique_percentage": float(col_data.nunique() / len(df) * 100),
                "inferred_type": self._infer_semantic_type(col_data),
                "notes": self._generate_column_notes(col_data)
            }
            
            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update(self._analyze_numeric_column(col_data))
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_info.update(self._analyze_datetime_column(col_data))
            else:
                col_info.update(self._analyze_categorical_column(col_data))
            
            column_analysis.append(col_info)
        
        return column_analysis
    
    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column beyond pandas dtype."""
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                if series.nunique() < 20 and series.min() >= 0:
                    return "categorical_numeric"
                return "integer"
            return "numeric"
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        elif pd.api.types.is_object_dtype(series):
            # Try to detect specific patterns
            if series.nunique() < 20:
                return "categorical"
            
            # Check if it could be dates
            try:
                pd.to_datetime(series.dropna().head(100), errors='raise')
                return "potential_datetime"
            except:
                pass
            
            # Check if it could be numeric
            try:
                pd.to_numeric(series.dropna().head(100), errors='raise')
                return "potential_numeric"
            except:
                pass
            
            # Check average length to distinguish between IDs and text
            avg_length = series.dropna().str.len().mean()
            if avg_length < 10:
                return "identifier"
            elif avg_length > 50:
                return "text"
            
            return "string"
        
        return "unknown"
    
    def _generate_column_notes(self, series: pd.Series) -> List[str]:
        """Generate notes about potential issues or characteristics."""
        notes = []
        
        # Missing data warning
        null_pct = series.isnull().sum() / len(series) * 100
        if null_pct > 50:
            notes.append("High missing data (>50%)")
        elif null_pct > 20:
            notes.append("Moderate missing data (>20%)")
        
        # Uniqueness notes
        unique_pct = series.nunique() / len(series) * 100
        if unique_pct > 95:
            notes.append("Likely unique identifier")
        elif unique_pct < 5 and series.nunique() < 10:
            notes.append("Low cardinality")
        
        # Type-specific notes
        if pd.api.types.is_object_dtype(series):
            if series.str.contains(r'^[0-9]+$', na=False).any():
                notes.append("Contains numeric strings")
            if series.str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
                notes.append("Contains date-like strings")
        
        # Constant values
        if series.nunique() == 1:
            notes.append("Constant value")
        
        return notes
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column specifics."""
        return {
            "min_value": float(series.min()) if not series.empty else None,
            "max_value": float(series.max()) if not series.empty else None,
            "mean_value": float(series.mean()) if not series.empty else None,
            "median_value": float(series.median()) if not series.empty else None,
            "std_deviation": float(series.std()) if not series.empty else None,
            "has_outliers": self._detect_outliers(series)
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column specifics."""
        return {
            "min_date": series.min().isoformat() if not series.empty else None,
            "max_date": series.max().isoformat() if not series.empty else None,
            "date_range_days": (series.max() - series.min()).days if not series.empty else None
        }
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column specifics."""
        value_counts = series.value_counts()
        return {
            "top_values": value_counts.head(5).to_dict(),
            "most_frequent": value_counts.index[0] if not value_counts.empty else None,
            "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else None
        }
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Simple outlier detection using IQR method."""
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            return False
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def _get_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        return {
            "total_missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / df.shape[0] * 100),
            "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
            "categorical_columns": int(df.select_dtypes(include=['object']).shape[1]),
            "datetime_columns": int(df.select_dtypes(include=['datetime64']).shape[1])
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality and flag issues."""
        issues = []
        quality_score = 100.0
        
        # Check for high missing data
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_pct > 30:
            issues.append("High overall missing data")
            quality_score -= 20
        elif missing_pct > 10:
            issues.append("Moderate overall missing data")
            quality_score -= 10
        
        # Check for duplicate rows
        dup_pct = df.duplicated().sum() / df.shape[0] * 100
        if dup_pct > 10:
            issues.append("High duplicate rows")
            quality_score -= 15
        elif dup_pct > 5:
            issues.append("Some duplicate rows")
            quality_score -= 5
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns: {', '.join(constant_cols)}")
            quality_score -= 10
        
        # Check for potential type issues
        potential_type_issues = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors='raise')
                    potential_type_issues += 1
                except:
                    pass
        
        if potential_type_issues > 0:
            issues.append(f"{potential_type_issues} columns may need type conversion")
            quality_score -= potential_type_issues * 5
        
        return {
            "quality_score": max(0, quality_score),
            "issues": issues,
            "recommendations": self._generate_recommendations(df, issues)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, issues: List[str]) -> List[str]:
        """Generate actionable recommendations based on data analysis."""
        recommendations = []
        
        if "High overall missing data" in ' '.join(issues):
            recommendations.append("Consider data imputation or removal of incomplete records")
        
        if "High duplicate rows" in ' '.join(issues):
            recommendations.append("Remove duplicate rows to avoid bias in analysis")
        
        if "Constant columns" in ' '.join(issues):
            recommendations.append("Remove constant columns as they provide no information")
        
        if any("may need type conversion" in issue for issue in issues):
            recommendations.append("Review column data types and convert as appropriate")
        
        # Add general recommendations
        if df.shape[1] > 20:
            recommendations.append("Consider feature selection for large number of columns")
        
        if df.shape[0] < 100:
            recommendations.append("Small dataset - be cautious with statistical inferences")
        
        return recommendations
    
    def _get_preview_data(self, df: pd.DataFrame, n_rows: int = 5) -> Dict[str, Any]:
        """Get preview data for display."""
        return {
            "head": df.head(n_rows).to_dict('records'),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    
    def save_metadata(self, analysis: Dict[str, Any], project_path: Path) -> bool:
        """
        Save dataset metadata to project directory.
        
        Args:
            analysis: Analysis results dictionary
            project_path: Path to the project directory
            
        Returns:
            True if saved successfully
        """
        try:
            metadata_path = project_path / "dataset_summary.json"
            
            # Make the analysis JSON serializable
            serializable_analysis = self._make_json_serializable(analysis)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def _make_json_serializable(self, obj):
        """Convert analysis results to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_summary_text(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the dataset analysis."""
        shape = analysis['shape']
        overall = analysis['overall_stats']
        quality = analysis['data_quality']
        
        summary = f"""Dataset: {analysis['filename']}

**Overview:**
- Shape: {shape['rows']:,} rows × {shape['columns']} columns
- Memory usage: {analysis['memory_usage_mb']:.1f} MB
- Data quality score: {quality['quality_score']:.1f}/100

**Column Types:**
- Numeric: {overall['numeric_columns']}
- Categorical: {overall['categorical_columns']}
- DateTime: {overall['datetime_columns']}

**Data Quality:**
- Missing values: {overall['total_missing_values']:,} ({overall['missing_percentage']:.1f}%)
- Duplicate rows: {overall['duplicate_rows']:,} ({overall['duplicate_percentage']:.1f}%)"""

        if quality['issues']:
            summary += f"\n\n**Issues Found:**\n" + "\n".join(f"- {issue}" for issue in quality['issues'])
        
        if quality['recommendations']:
            summary += f"\n\n**Recommendations:**\n" + "\n".join(f"- {rec}" for rec in quality['recommendations'])
        
        return summary