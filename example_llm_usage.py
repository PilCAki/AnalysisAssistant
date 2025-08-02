#!/usr/bin/env python3
"""
Example usage of the LLM interface for data analysis.

This script demonstrates how to use the LLM interface with a real OpenAI API key
to generate analysis code and track interactions.

Usage:
    export OPENAI_API_KEY="your-api-key-here"
    python example_llm_usage.py
"""

import os
from data_analysis_assistant.agent_core import AnalysisEngine, AssistantResponse
from data_analysis_assistant.agent_core.persistence import PersistenceManager
from data_analysis_assistant.agent_core.summary_generator import SessionSummarizer


def main():
    """Example of using the LLM interface for analysis."""
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 LLM Interface Example")
    print("=" * 40)
    
    # Create or use existing project
    persistence = PersistenceManager()
    
    try:
        # Try to create a new project
        project_info = persistence.create_project("example_analysis")
        print(f"✅ Created new project: {project_info['project_name']}")
    except ValueError as e:
        if "already exists" in str(e):
            print("ℹ️ Using existing project: example_analysis")
        else:
            raise
    
    # Initialize analysis engine
    engine = AnalysisEngine(project_name="example_analysis")
    
    # Example context for a sales dataset
    context = {
        'dataset_name': 'sales_data.csv',
        'column_info': 'date, product, sales_amount, region, customer_id',
        'data_shape': '(10000, 5)',
        'data_types': 'datetime, str, float, str, int',
        'conventions': 'Use descriptive variable names, add comments, use matplotlib for visualizations'
    }
    
    # Example analysis requests
    requests = [
        "Plot average sales by region over time using matplotlib",
        "Show correlation between sales and customer demographics",
        "Create a monthly sales trend analysis with seasonality detection"
    ]
    
    print(f"\n📊 Running {len(requests)} analysis requests...")
    
    responses = []
    for i, request in enumerate(requests, 1):
        print(f"\n🔍 Request {i}: {request}")
        
        try:
            response = engine.plan_analysis(request, context=context)
            responses.append(response)
            
            print(f"✅ Generated response:")
            print(f"   📝 Explanation: {response.explanation[:100]}...")
            print(f"   💻 Code: {len(response.code)} characters")
            if response.visualization_type:
                print(f"   📊 Visualization: {response.visualization_type}")
            if response.assumptions:
                print(f"   🔍 Assumptions: {len(response.assumptions)} items")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Generate session summary
    if responses:
        print(f"\n📄 Generating session summary...")
        try:
            summarizer = SessionSummarizer("projects/example_analysis")
            summary_path = summarizer.create_session_summary()
            print(f"✅ Summary saved to: {summary_path}")
            
            # Show summary snippet
            with open(summary_path, 'r') as f:
                lines = f.readlines()[:15]
            print("\n📋 Summary preview:")
            print("".join(lines))
            print("...")
            
        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
    
    print(f"\n🎉 Example completed! Generated {len(responses)} analysis responses.")
    print("\n💡 Check the following files:")
    print("   📄 projects/example_analysis/llm_log_*.jsonl - Interaction logs")
    print("   📋 projects/example_analysis/llm_summary.md - Session summary")


if __name__ == "__main__":
    main()