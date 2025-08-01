#!/usr/bin/env python3
"""
Simple demo script for LLM interface functionality.

This script demonstrates core functionality without requiring an API key.
"""

import os
import tempfile
import json
from pathlib import Path

from data_analysis_assistant.agent_core.models import AssistantResponse, LLMConfig, LLMLogEntry
from data_analysis_assistant.agent_core.persistence import PersistenceManager
from data_analysis_assistant.agent_core.summary_generator import SessionSummarizer
from data_analysis_assistant.agent_core.engine import LLMInterface


def demo_models_and_logging():
    """Demonstrate model creation and logging functionality."""
    print("🏗️ Model and Logging Demo")
    print("=" * 40)
    
    # Create a sample AssistantResponse
    response = AssistantResponse(
        explanation="I'll analyze the sales data and create a visualization showing trends over time.",
        code="""import matplotlib.pyplot as plt
import pandas as pd

# Load and analyze the data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Create the visualization
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['sales'], marker='o', linewidth=2)
plt.title('Sales Trends Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
        assumptions=["Sales data is clean and complete", "Date format is consistent"],
        next_steps="Consider segmenting by region or product category",
        visualization_type="line plot"
    )
    
    print("✅ Created structured response:")
    print(f"   📝 Explanation: {len(response.explanation)} chars")
    print(f"   💻 Code: {len(response.code.split('\\n'))} lines")
    print(f"   🔍 Assumptions: {len(response.assumptions)}")
    print(f"   📊 Viz type: {response.visualization_type}")
    
    # Create a log entry
    log_entry = LLMLogEntry(
        prompt_id="demo-12345",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        user_prompt="Create a line plot showing sales trends over time",
        response_text="I'll analyze the sales data and create a visualization...",
        structured_response=response.model_dump(),
        tokens_used=245,
        duration_seconds=2.3
    )
    
    print("\\n✅ Created log entry:")
    print(f"   🆔 ID: {log_entry.prompt_id}")
    print(f"   🤖 Model: {log_entry.model}")
    print(f"   🎛️ Temperature: {log_entry.temperature}")
    print(f"   🎯 Tokens: {log_entry.tokens_used}")
    print(f"   ⏱️ Duration: {log_entry.duration_seconds}s")
    
    return response, log_entry


def demo_project_and_logging():
    """Demonstrate project creation and log file management."""
    print("\\n📁 Project and Logging Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create project
        persistence = PersistenceManager(temp_dir)
        project_info = persistence.create_project("sales_analysis")
        
        print(f"✅ Created project: {project_info['project_name']}")
        print(f"   📂 Path: {project_info['project_path']}")
        
        # Simulate multiple LLM interactions
        interactions = [
            {
                "prompt_id": "req-001", 
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 500,
                "user_prompt": "Load and explore the sales dataset",
                "response_text": "I'll help you load and explore the sales data.",
                "structured_response": {
                    "explanation": "Loading the sales dataset for initial exploration",
                    "code": "import pandas as pd\\ndf = pd.read_csv('sales.csv')\\ndf.info()",
                    "assumptions": ["Data is in CSV format"],
                    "visualization_type": None
                },
                "tokens_used": 120,
                "duration_seconds": 1.8
            },
            {
                "prompt_id": "req-002",
                "model": "gpt-4o-mini", 
                "temperature": 0.7,
                "max_tokens": 800,
                "user_prompt": "Create a sales trend visualization",
                "response_text": "I'll create a line plot showing sales trends over time.",
                "structured_response": {
                    "explanation": "Creating a time series visualization of sales data",
                    "code": "plt.figure(figsize=(10,6))\\nplt.plot(df['date'], df['sales'])\\nplt.title('Sales Trends')\\nplt.show()",
                    "assumptions": ["Date column exists", "Sales data is numeric"],
                    "visualization_type": "line plot"
                },
                "tokens_used": 180,
                "duration_seconds": 2.1
            },
            {
                "prompt_id": "req-003",
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 600,
                "user_prompt": "Show sales by region",
                "response_text": "I'll create a bar chart showing sales by region.",
                "structured_response": {
                    "explanation": "Analyzing sales performance across different regions",
                    "code": "df.groupby('region')['sales'].sum().plot(kind='bar')\\nplt.title('Sales by Region')\\nplt.show()",
                    "assumptions": ["Region column exists"],
                    "visualization_type": "bar chart"
                },
                "tokens_used": 165,
                "duration_seconds": 1.9
            }
        ]
        
        # Log all interactions
        for interaction in interactions:
            persistence.log_llm_interaction(project_info['project_name'], interaction)
        
        print(f"✅ Logged {len(interactions)} interactions")
        
        # Check log files
        log_files = list(Path(project_info['project_path']).glob("llm_log_*.jsonl"))
        print(f"   📄 Log files: {len(log_files)}")
        
        if log_files:
            with open(log_files[0], 'r') as f:
                entries = [json.loads(line) for line in f]
            print(f"   📊 Log entries: {len(entries)}")
            
            # Generate summary
            summarizer = SessionSummarizer(project_info['project_path'])
            summary_path = summarizer.create_session_summary()
            print(f"   📋 Summary: {Path(summary_path).name}")
            
            # Show summary content
            with open(summary_path, 'r') as f:
                content = f.read()
                lines = content.split('\\n')[:20]  # First 20 lines
                
            print("\\n📋 Summary preview:")
            print("\\n".join(lines))
            print("...")
            
            return project_info['project_path'], len(entries)


def demo_configuration():
    """Demonstrate configuration handling."""
    print("\\n🔧 Configuration Demo")
    print("=" * 30)
    
    # Test config without API key
    with_api_key = os.environ.get('OPENAI_API_KEY') is not None
    
    config = LLMConfig()
    print(f"✅ Default model: {config.default_model}")
    print(f"✅ Default temperature: {config.default_temperature}")
    print(f"✅ Max retries: {config.max_retries}")
    print(f"✅ API key configured: {config.is_configured}")
    
    if with_api_key:
        print("✅ Ready for real API calls")
    else:
        print("ℹ️ No API key - demo mode only")
    
    # Test LLM interface initialization
    llm = LLMInterface()
    print(f"✅ LLM interface created")
    print(f"   🔗 Client available: {llm.client is not None}")


def demo_prompt_formatting():
    """Demonstrate prompt formatting functionality."""
    print("\\n💬 Prompt Formatting Demo")
    print("=" * 35)
    
    llm = LLMInterface()
    
    # Test basic prompt
    messages = llm.format_prompt("Analyze this dataset")
    print(f"✅ Basic prompt: {len(messages)} messages")
    
    # Test with context
    context = {
        'dataset_name': 'customer_data.csv',
        'column_info': 'customer_id, age, income, region, purchases',
        'data_shape': '(5000, 5)',
        'data_types': 'int, int, float, str, int',
        'conventions': 'Use descriptive variable names and add comments'
    }
    
    messages_with_context = llm.format_prompt(
        "Create a customer segmentation analysis", 
        context=context
    )
    print(f"✅ Context prompt: {len(messages_with_context)} messages")
    
    # Show system prompt content (first 200 chars)
    if messages_with_context and messages_with_context[0]['role'] == 'system':
        system_content = messages_with_context[0]['content'][:200]
        print(f"   📝 System prompt: {system_content}...")
    
    return len(messages), len(messages_with_context)


if __name__ == "__main__":
    print("🚀 LLM Interface Demonstration")
    print("=" * 50)
    print("This demo shows the LLM interface capabilities without requiring an API key.\\n")
    
    try:
        # Run all demos
        response, log_entry = demo_models_and_logging()
        project_path, num_entries = demo_project_and_logging()
        demo_configuration()
        basic_msgs, context_msgs = demo_prompt_formatting()
        
        print("\\n🎉 Demo completed successfully!")
        print("\\n📊 Summary:")
        print(f"   ✅ Structured response model: {len(response.code)} chars of code")
        print(f"   ✅ Log entry created: {log_entry.tokens_used} tokens")
        print(f"   ✅ Project with {num_entries} logged interactions")
        print(f"   ✅ Prompt formatting: {basic_msgs} → {context_msgs} messages")
        
        print("\\n🏆 Key features demonstrated:")
        print("   ✅ Pydantic models for structured responses")
        print("   ✅ Comprehensive interaction logging to JSONL")
        print("   ✅ Automatic session summary generation")
        print("   ✅ Project integration and management")
        print("   ✅ Context-aware prompt formatting")
        print("   ✅ Configuration and error handling")
        
        if not os.environ.get('OPENAI_API_KEY'):
            print("\\n💡 To test with real API calls:")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            print("   python demo_simple.py")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise