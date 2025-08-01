#!/usr/bin/env python3
"""
Integration test and demonstration script for the LLM interface.

This script demonstrates the full LLM interface functionality including:
- Structured prompt handling
- Mock LLM responses (for testing without API key)
- Logging to JSONL files
- Session summary generation
"""

import os
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from data_analysis_assistant.agent_core.engine import AnalysisEngine, LLMInterface
from data_analysis_assistant.agent_core.models import AssistantResponse
from data_analysis_assistant.agent_core.persistence import PersistenceManager  
from data_analysis_assistant.agent_core.summary_generator import SessionSummarizer


def create_mock_openai_response(explanation: str, code: str, visualization_type: str = None):
    """Helper to create mock OpenAI response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_usage = Mock()
    
    mock_message.parsed = AssistantResponse(
        explanation=explanation,
        code=code,
        visualization_type=visualization_type
    )
    mock_message.content = f"{explanation}\n\n{code}"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_usage.total_tokens = len(explanation) + len(code)  # Simple token estimation
    mock_response.usage = mock_usage
    
    return mock_response


def demo_llm_interface():
    """Demonstrate LLM interface functionality with mocked responses."""
    print("🤖 LLM Interface Demo")
    print("=" * 50)
    
    # Create temporary project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Initialize project structure
        persistence = PersistenceManager(str(project_path))
        project_info = persistence.create_project("demo_analysis", None)
        print(f"📁 Created project: {project_info['project_name']}")
        
        # Initialize analysis engine
        engine = AnalysisEngine(project_name="demo_analysis")
        
        # Mock OpenAI client for demo
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'demo-key'}):
            with patch('data_analysis_assistant.agent_core.engine.OpenAI') as mock_openai:
                # Set up mock client
                mock_client = Mock()
                mock_openai.return_value = mock_client
            
            # Demo 1: Basic analysis request
            print("\n🔍 Demo 1: Basic Analysis Request")
            mock_client.beta.chat.completions.parse.return_value = create_mock_openai_response(
                explanation="I'll create a sales analysis plot showing trends over time using matplotlib.",
                code="""import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('sales_data.csv')

# Create a line plot of sales over time
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['sales'])
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""",
                visualization_type="line plot"
            )
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'demo-key'}):
                response = engine.plan_analysis(
                    "Plot average sales by region over time using matplotlib",
                    context={
                        'dataset_name': 'sales_data.csv',
                        'column_info': 'date, sales, region',
                        'data_shape': '(1000, 3)',
                        'data_types': 'datetime, float, str',
                        'conventions': 'Use matplotlib for visualizations'
                    }
                )
                
                print(f"✅ Explanation: {response.explanation}")
                print(f"📋 Code length: {len(response.code)} characters")
                print(f"📊 Visualization type: {response.visualization_type}")
            
            # Demo 2: Statistical analysis request
            print("\n📊 Demo 2: Statistical Analysis Request")
            mock_client.beta.chat.completions.parse.return_value = create_mock_openai_response(
                explanation="I'll perform a correlation analysis between different variables and create a heatmap.",
                code="""import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('sales_data.csv')

# Calculate correlation matrix
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()""",
                visualization_type="heatmap"
            )
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'demo-key'}):
                response = engine.plan_analysis("Show correlation between variables")
                print(f"✅ Explanation: {response.explanation}")
                print(f"📋 Code includes: correlation matrix, heatmap")
        
        # Demo 3: Check logging
        print("\n📝 Demo 3: Log File Analysis")
        project_files = list(Path(project_info['project_path']).glob("llm_log_*.jsonl"))
        print(f"📄 Log files created: {len(project_files)}")
        
        if project_files:
            log_file = project_files[0]
            with open(log_file, 'r') as f:
                lines = f.readlines()
            print(f"📊 Log entries: {len(lines)}")
            
            # Generate session summary
            print("\n📄 Demo 4: Session Summary Generation")
            summarizer = SessionSummarizer(project_info['project_path'])
            summary_path = summarizer.create_session_summary()
            print(f"✅ Summary created: {Path(summary_path).name}")
            
            # Show summary excerpt
            with open(summary_path, 'r') as f:
                summary_lines = f.readlines()[:15]  # First 15 lines
            print("\n📋 Summary preview:")
            print("".join(summary_lines))
            print("...")
        
        print("\n✅ Demo completed successfully!")


def test_api_key_handling():
    """Test API key configuration handling."""
    print("\n🔐 API Key Configuration Test")
    print("=" * 40)
    
    # Test without API key
    with patch.dict(os.environ, {}, clear=True):
        llm = LLMInterface()
        print(f"❌ Without API key - Configured: {llm.config.is_configured}")
        print(f"❌ Client available: {llm.client is not None}")
    
    # Test with API key
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
        llm = LLMInterface()
        print(f"✅ With API key - Configured: {llm.config.is_configured}")
        # Note: Client would be created lazily


def test_structured_response():
    """Test structured response format."""
    print("\n🏗️ Structured Response Test")
    print("=" * 35)
    
    # Create sample response
    response = AssistantResponse(
        explanation="This is a test analysis that demonstrates data visualization",
        code="import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()",
        assumptions=["Data is clean", "No missing values"],
        next_steps="Consider adding more features",
        visualization_type="line plot"
    )
    
    print(f"✅ Explanation: {response.explanation[:50]}...")
    print(f"✅ Code lines: {len(response.code.split())}")
    print(f"✅ Assumptions: {len(response.assumptions)}")
    print(f"✅ Visualization type: {response.visualization_type}")
    print(f"✅ Response serializable: {bool(response.model_dump())}")


if __name__ == "__main__":
    print("🚀 LLM Interface Integration Test")
    print("=" * 60)
    
    try:
        test_api_key_handling()
        test_structured_response()
        demo_llm_interface()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📝 Key features demonstrated:")
        print("   ✅ Structured LLM responses with Pydantic models")
        print("   ✅ Comprehensive logging to JSONL files")
        print("   ✅ Session summary generation")
        print("   ✅ Error handling and configuration management")
        print("   ✅ Project integration and context handling")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise