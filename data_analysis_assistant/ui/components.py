"""
UI components for the AnalysisAssistant Streamlit interface.

This module contains reusable UI components for chat interface,
data visualization, and project management.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List


class ChatComponents:
    """Reusable chat interface components."""
    
    @staticmethod
    def display_message(role: str, content: str, code: str = None, 
                       result: Any = None, plot: plt.Figure = None):
        """
        Display a chat message with optional code and results.
        """
        with st.chat_message(role):
            st.write(content)
            
            # Display code if present
            if code:
                st.subheader("Generated Code:")
                st.code(code, language="python")
            
            # Display results if present
            if result:
                st.subheader("Results:")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, dict):
                    st.json(result)
                else:
                    st.write(result)
            
            # Display plot if present
            if plot:
                st.pyplot(plot)
    
    @staticmethod
    def code_viewer(code: str, language: str = "python", 
                   editable: bool = False) -> str:
        """
        Display code with syntax highlighting and optional editing.
        """
        if editable:
            return st.text_area("Edit Code:", value=code, height=200)
        else:
            st.code(code, language=language)
            return code
    
    @staticmethod
    def result_viewer(result: Any, result_type: str = "auto"):
        """
        Display analysis results in appropriate format.
        """
        if result is None:
            return
            
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, dict):
            st.json(result)
        elif isinstance(result, str) and len(result) > 100:
            st.text_area("Result:", value=result, height=150, disabled=True)
        else:
            st.write(result)


class DataComponents:
    """Components for data display and interaction."""
    
    @staticmethod
    def data_preview(df: pd.DataFrame, max_rows: int = 100):
        """
        Display interactive data preview with basic information.
        """
        if df is None:
            st.info("No data uploaded yet.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        
        with col2:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Data types summary
        st.subheader("Data Types")
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
        
        # Data preview
        st.subheader("Data Preview")
        display_rows = min(max_rows, len(df))
        st.dataframe(df.head(display_rows), use_container_width=True)
        
        if len(df) > display_rows:
            st.info(f"Showing first {display_rows} of {len(df)} rows")
    
    @staticmethod
    def comprehensive_dataset_preview(analysis: Dict[str, Any]):
        """
        Display comprehensive dataset analysis in an expandable section.
        
        Args:
            analysis: Dataset analysis results from DatasetAnalyzer
        """
        with st.expander("📊 Dataset Analysis & Preview", expanded=True):
            # Overview metrics
            shape = analysis['shape']
            overall = analysis['overall_stats']
            quality = analysis['data_quality']
            
            st.markdown(f"**Dataset:** {analysis['filename']}")
            st.markdown(f"**Shape:** {shape['rows']:,} rows × {shape['columns']} columns")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Memory Usage", f"{analysis['memory_usage_mb']:.1f} MB")
            with col2:
                st.metric("Missing Values", f"{overall['total_missing_values']:,}")
            with col3:
                st.metric("Duplicate Rows", f"{overall['duplicate_rows']:,}")
            with col4:
                st.metric("Quality Score", f"{quality['quality_score']:.0f}/100")
            
            # Column Summary Table
            st.subheader("Column Summary")
            column_data = []
            for col_info in analysis['column_summary']:
                notes = col_info.get('notes', [])
                notes_str = "; ".join(notes) if notes else "-"
                
                row = {
                    "Column": col_info['name'],
                    "Type": col_info['dtype'],
                    "Nulls": col_info['null_count'],
                    "Unique": col_info['unique_count'],
                    "Notes": notes_str[:50] + "..." if len(notes_str) > 50 else notes_str
                }
                column_data.append(row)
            
            column_df = pd.DataFrame(column_data)
            st.dataframe(column_df, use_container_width=True, hide_index=True)
            
            # Data Quality Issues
            if quality['issues']:
                st.subheader("⚠️ Data Quality Issues")
                for issue in quality['issues']:
                    st.warning(issue)
            
            # Recommendations
            if quality['recommendations']:
                st.subheader("💡 Recommendations")
                for rec in quality['recommendations']:
                    st.info(rec)
            
            # Data Preview (first few rows)
            st.subheader("Data Preview (First 5 Rows)")
            preview_data = analysis['preview_data']
            preview_df = pd.DataFrame(preview_data['head'])
            if not preview_df.empty:
                st.dataframe(preview_df, use_container_width=True)
            else:
                st.info("No data to preview")
    
    @staticmethod
    def column_selector(df: pd.DataFrame, 
                       multiselect: bool = False) -> List[str]:
        """
        Create column selection interface.
        """
        if df is None:
            return []
            
        columns = df.columns.tolist()
        
        if multiselect:
            return st.multiselect("Select columns:", columns)
        else:
            selected = st.selectbox("Select column:", columns)
            return [selected] if selected else []
    
    @staticmethod
    def data_info_panel(df: pd.DataFrame):
        """
        Display comprehensive data information panel.
        """
        if df is None:
            st.info("No data loaded")
            return
            
        with st.expander("📊 Dataset Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).shape[1]
                categorical_cols = df.select_dtypes(include=['object']).shape[1]
                st.metric("Numeric Columns", numeric_cols)
                st.metric("Text Columns", categorical_cols)
            
            with col3:
                missing_count = df.isnull().sum().sum()
                duplicate_count = df.duplicated().sum()
                st.metric("Missing Values", missing_count)
                st.metric("Duplicate Rows", duplicate_count)
            
            # Column details
            st.subheader("Column Details")
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": df[col].count(),
                    "Missing": df[col].isnull().sum(),
                    "Unique": df[col].nunique()
                })
            
            info_df = pd.DataFrame(col_info)
            st.dataframe(info_df, use_container_width=True)


class ProjectComponents:
    """Components for project management."""
    
    @staticmethod
    def project_selector(projects: List[str], current_project: Optional[str] = None) -> Optional[str]:
        """
        Create project selection interface.
        """
        st.subheader("🗂️ Project Management")
        
        # Project selection
        if projects:
            current_index = 0
            if current_project and current_project in projects:
                current_index = projects.index(current_project)
            
            selected_project = st.selectbox(
                "Select Project:",
                options=[""] + projects,
                index=current_index + 1 if current_project else 0
            )
            
            if selected_project and selected_project != current_project:
                return selected_project
        else:
            st.info("No projects found. Create a new project below.")
        
        # New project creation
        st.subheader("Create New Project")
        new_project_name = st.text_input(
            "Project Name:",
            placeholder="my_analysis_project"
        )
        
        if st.button("Create Project", disabled=not new_project_name):
            return f"CREATE:{new_project_name}"
        
        return None
    
    @staticmethod
    def project_info_display(project_name: Optional[str], project_path: Optional[str] = None):
        """
        Display current project information.
        """
        if project_name:
            st.success(f"📁 Current Project: **{project_name}**")
            if project_path:
                st.caption(f"Location: {project_path}")
        else:
            st.info("👆 Please select or create a project to get started")


class SettingsComponents:
    """Components for application settings."""
    
    @staticmethod
    def model_settings():
        """
        Interface for LLM model settings.
        """
        with st.expander("🔧 LLM Settings"):
            st.info("LLM configuration will be available in future updates.")
            
            # Placeholder for future settings
            st.text_input("OpenAI API Key", type="password", disabled=True)
            st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], disabled=True)
            st.slider("Temperature", 0.0, 1.0, 0.7, disabled=True)
    
    @staticmethod
    def analysis_preferences():
        """
        Interface for analysis preferences.
        """
        with st.expander("⚙️ Analysis Preferences"):
            st.checkbox("Auto-execute generated code", value=False)
            st.checkbox("Show code by default", value=True)
            st.selectbox("Plot style", ["default", "seaborn", "ggplot"])
            st.number_input("Max rows to display", min_value=10, max_value=1000, value=100)