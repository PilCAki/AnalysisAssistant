"""
Usage examples for the ConventionManager.

This demonstrates how to integrate the ConventionManager with the Analysis Assistant.
"""

import pandas as pd
from pathlib import Path
from data_analysis_assistant.agent_core.conventions import ConventionManager


def example_basic_usage():
    """Example of basic ConventionManager usage."""
    print("=== Basic ConventionManager Usage ===")
    
    # Create a sample project
    project_path = Path("/tmp/example_project")
    project_path.mkdir(exist_ok=True)
    
    # Initialize Convention Manager
    cm = ConventionManager(project_path)
    
    # Set some conventions manually
    cm.set("time_column", "timestamp")
    cm.set("id_column", "customer_id")
    
    # Save conventions
    cm.save()
    
    print(f"Conventions saved to: {cm.conventions_file}")
    print(f"Time column: {cm.get('time_column')}")
    print(f"ID column: {cm.get('id_column')}")


def example_data_inference():
    """Example of automatic convention inference from data."""
    print("\n=== Data Inference Example ===")
    
    # Create sample dataset
    df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "age": [25, 30, 35, 40, 45],
        "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "region": ["North", "South", "East", "West", "North"],
        "churn": [0, 1, 0, 1, 0]
    })
    
    print("Sample dataset:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Column types:\n{df.dtypes}")
    
    # Create Convention Manager
    project_path = Path("/tmp/inference_project")
    project_path.mkdir(exist_ok=True)
    cm = ConventionManager(project_path)
    
    # Infer conventions from data
    inferred = cm.infer_from_data(df)
    
    print(f"\nInferred conventions: {inferred}")
    print(f"Time column: {cm.get('time_column')}")
    print(f"ID column: {cm.get('id_column')}")
    print(f"Target column: {cm.get('target_column')}")
    print(f"Numeric columns: {cm.get('numeric_columns')}")
    print(f"Category columns: {cm.get('category_columns')}")
    print(f"Date format: {cm.get('date_format')}")


def example_user_prompting():
    """Example of user prompting for missing conventions."""
    print("\n=== User Prompting Example ===")
    
    # Create Convention Manager with partial data
    project_path = Path("/tmp/prompting_project")
    project_path.mkdir(exist_ok=True)
    cm = ConventionManager(project_path)
    
    # Set some conventions, leave others empty
    cm.set("time_column", "date")
    
    # Define UI callback for prompting
    def mock_ui_callback(prompt, description, current_value):
        """Mock UI callback that simulates user responses."""
        print(f"PROMPT: {prompt}")
        print(f"DESCRIPTION: {description}")
        if current_value:
            print(f"CURRENT VALUE: {current_value}")
        
        # Simulate user responses
        if "id" in prompt.lower():
            return "user_id"
        elif "target" in prompt.lower():
            return "conversion"
        return ""
    
    # Prompt for missing conventions
    responses = cm.prompt_missing_conventions(mock_ui_callback)
    
    print(f"\nUser responses: {responses}")
    print(f"Final conventions: {cm.conventions}")


def example_integration_with_project():
    """Example showing integration with project structure."""
    print("\n=== Project Integration Example ===")
    
    # This would typically be called during project initialization
    project_path = Path("/tmp/integrated_project")
    project_path.mkdir(exist_ok=True)
    
    # Create Convention Manager
    cm = ConventionManager(project_path)
    
    # Load sample data
    df = pd.DataFrame({
        "transaction_id": [f"T{i:04d}" for i in range(1, 11)],
        "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
        "amount": [100.0 + i*10 for i in range(10)],
        "category": ["A", "B", "A", "B", "C"] * 2,
        "is_fraud": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    })
    
    print("Dataset for analysis:")
    print(df.head())
    
    # Infer conventions
    inferred = cm.infer_from_data(df)
    print(f"\nInferred conventions: {inferred}")
    
    # Save conventions for use by other components
    cm.save()
    print(f"Conventions saved for project use")
    
    # Demonstrate how other components would load conventions
    new_cm = ConventionManager(project_path)
    print(f"Loaded time column: {new_cm.get('time_column')}")
    print(f"Loaded ID column: {new_cm.get('id_column')}")
    
    # Example: Use conventions for LLM prompt generation
    prompt_context = f"""
    Dataset Analysis Context:
    - Time column: {new_cm.get('time_column')}
    - ID column: {new_cm.get('id_column')}
    - Target column: {new_cm.get('target_column')}
    - Numeric columns: {new_cm.get('numeric_columns')}
    - Category columns: {new_cm.get('category_columns')}
    - Date format: {new_cm.get('date_format')}
    
    This context will help the LLM understand the dataset structure.
    """
    print(f"\nLLM Prompt Context:\n{prompt_context}")


if __name__ == "__main__":
    example_basic_usage()
    example_data_inference()
    example_user_prompting()
    example_integration_with_project()