"""
Tests for the prompt → code generation loop functionality.

This module tests the complete implementation of the requirements specified in Issue #15:
- Prompt formatting with context (dataset metadata, conventions)
- OpenAI API calls with structured response parsing
- Response parsing into plan and executable code
- JSONL logging in the exact format specified
- Configurable parameters (no hardcoded values)
- Support for multiple datasets and prompt types
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from data_analysis_assistant.agent_core.engine import LLMInterface, AnalysisEngine
from data_analysis_assistant.agent_core.models import AssistantResponse, LLMConfig
from data_analysis_assistant.agent_core.persistence import PersistenceManager


class TestPromptToCodeLoop:
    """Test the complete prompt → code generation loop implementation."""
    
    def test_prompt_formatting_with_context(self):
        """Test that prompts are formatted correctly with dataset context."""
        llm = LLMInterface()
        
        context = {
            'dataset_name': 'test_data.csv',
            'column_info': 'id, name, value',
            'data_shape': '(100, 3)',
            'data_types': 'int, str, float',
            'conventions': 'Use pandas for data manipulation'
        }
        
        user_request = 'Analyze the data'
        messages = llm.format_prompt(user_request, context)
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'system'
        assert 'test_data.csv' in messages[0]['content']
        assert 'id, name, value' in messages[0]['content']
        assert 'Use pandas for data manipulation' in messages[0]['content']
        assert 'Analyze the data' in messages[0]['content']
    
    def test_structured_response_parsing(self):
        """Test that responses are properly parsed into plan and code."""
        response = AssistantResponse(
            explanation="I will analyze the data by loading it and creating a summary.",
            code="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())",
            assumptions=["Data is clean", "CSV format is valid"],
            next_steps="Create visualizations",
            visualization_type="summary_table"
        )
        
        # Verify structure
        assert isinstance(response.explanation, str)
        assert isinstance(response.code, str)
        assert isinstance(response.assumptions, list)
        assert len(response.explanation) > 0
        assert len(response.code) > 0
        assert 'import pandas as pd' in response.code
    
    def test_llm_config_no_hardcoded_parameters(self):
        """Test that LLM configuration uses configurable parameters, not hardcoded values."""
        config = LLMConfig()
        
        # All parameters should be configurable
        assert hasattr(config, 'default_model')
        assert hasattr(config, 'default_temperature')
        assert hasattr(config, 'default_max_tokens')
        assert hasattr(config, 'timeout')
        assert hasattr(config, 'max_retries')
        
        # Should use reasonable defaults
        assert config.default_model == "gpt-4o-mini"
        assert 0.0 <= config.default_temperature <= 2.0
        assert config.default_max_tokens > 0
        assert config.timeout > 0
        assert config.max_retries > 0
    
    def test_jsonl_logging_format(self):
        """Test that JSONL logs match the exact format specified in the issue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            project_info = pm.create_project("test_logging")
            
            # Create test interaction in the exact format from the issue
            test_interaction = {
                'user_prompt': 'Plot revenue by month',
                'llm_raw_response': 'I will help you create a revenue plot...',
                'llm_plan': 'I will analyze monthly revenue trends and create a line plot',
                'llm_code': 'import matplotlib.pyplot as plt\nimport pandas as pd\n# Code here'
            }
            
            pm.log_llm_interaction('test_logging', test_interaction)
            
            # Verify log file exists and has correct format
            project_path = Path(tmpdir) / 'test_logging'
            log_files = list(project_path.glob('llm_log_*.jsonl'))
            assert len(log_files) == 1
            
            with open(log_files[0], 'r') as f:
                log_content = f.read().strip()
            
            log_entry = json.loads(log_content)
            
            # Verify exact format from issue specification
            required_fields = ['timestamp', 'user_prompt', 'llm_raw_response', 'llm_plan', 'llm_code']
            for field in required_fields:
                assert field in log_entry, f"Missing required field: {field}"
            
            assert log_entry['user_prompt'] == 'Plot revenue by month'
            assert log_entry['llm_plan'].startswith('I will analyze')
            assert 'import matplotlib.pyplot as plt' in log_entry['llm_code']
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_complete_prompt_to_code_loop(self, mock_openai_class):
        """Test the complete prompt → code generation loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            pm = PersistenceManager(tmpdir)
            project_info = pm.create_project("test_complete_loop")
            
            llm = LLMInterface(project_name="test_complete_loop")
            llm.persistence = pm
            
            # Mock response
            mock_response = AssistantResponse(
                explanation="I will create a revenue analysis plot.",
                code="import pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_csv('data.csv')\ndf.plot()",
                assumptions=["Data is valid"],
                visualization_type="line_plot"
            )
            
            # Mock OpenAI call
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = "Raw response content"
            mock_completion.choices[0].message.parsed = mock_response
            mock_completion.usage.total_tokens = 300
            
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_completion
            mock_openai_class.return_value = mock_client
            
            # Force recreate client
            llm._client = None
            llm.config.api_key = 'test-key'
            
            # Test context
            context = {
                'dataset_name': 'revenue.csv',
                'column_info': 'date, amount, region',
                'data_shape': '(1000, 3)',
                'data_types': 'datetime, float, str',
                'conventions': 'Use matplotlib for visualizations'
            }
            
            # Execute the complete loop
            response = llm.get_structured_response(
                "Plot revenue by month",
                context=context
            )
            
            # Verify response
            assert isinstance(response, AssistantResponse)
            assert response.explanation == "I will create a revenue analysis plot."
            assert "import pandas as pd" in response.code
            assert response.visualization_type == "line_plot"
            
            # Verify logging occurred
            project_path = Path(tmpdir) / "test_complete_loop"
            log_files = list(project_path.glob("llm_log_*.jsonl"))
            assert len(log_files) == 1
            
            with open(log_files[0], 'r') as f:
                log_entry = json.loads(f.read().strip())
            
            # Verify log format matches specification
            assert 'timestamp' in log_entry
            assert 'user_prompt' in log_entry
            assert 'llm_raw_response' in log_entry
            assert 'llm_plan' in log_entry
            assert 'llm_code' in log_entry
            
            assert log_entry['llm_plan'] == "I will create a revenue analysis plot."
            assert "import pandas as pd" in log_entry['llm_code']
    
    def test_multiple_datasets_and_prompts(self):
        """Test that the system works with multiple datasets and prompt types."""
        llm = LLMInterface()
        
        # Test different dataset contexts
        datasets = [
            {
                'dataset_name': 'sales.csv',
                'column_info': 'date, sales, region',
                'conventions': 'Use matplotlib'
            },
            {
                'dataset_name': 'customers.csv', 
                'column_info': 'id, age, satisfaction',
                'conventions': 'Use seaborn'
            },
            {
                'dataset_name': 'financial.csv',
                'column_info': 'quarter, revenue, expenses',
                'conventions': 'Use plotly'
            }
        ]
        
        prompts = [
            "Create a sales trend analysis",
            "Analyze customer satisfaction by age",
            "Build a financial performance dashboard"
        ]
        
        # Test that each combination produces valid formatted prompts
        for dataset in datasets:
            for prompt in prompts:
                messages = llm.format_prompt(prompt, dataset)
                
                assert len(messages) == 1
                assert messages[0]['role'] == 'system'
                assert dataset['dataset_name'] in messages[0]['content']
                assert prompt in messages[0]['content']
                assert dataset['conventions'] in messages[0]['content']
    
    def test_analysis_engine_integration(self):
        """Test that AnalysisEngine properly integrates with the LLM interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            project_info = pm.create_project("test_engine")
            
            engine = AnalysisEngine(project_name="test_engine")
            
            # Verify integration
            assert engine.project_name == "test_engine"
            assert isinstance(engine.llm, LLMInterface)
            assert engine.llm.project_name == "test_engine"
            assert isinstance(engine.persistence, PersistenceManager)
    
    def test_error_handling_and_logging(self):
        """Test that errors are properly handled and logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            project_info = pm.create_project("test_errors")
            
            llm = LLMInterface(project_name="test_errors")
            llm.persistence = pm
            
            # Test with no API key (should raise exception)
            with patch.dict(os.environ, {}, clear=True):
                llm._client = None
                llm.config.api_key = None
                
                with pytest.raises(Exception, match="OpenAI client not initialized"):
                    llm.call_llm([{"role": "user", "content": "test"}])


class TestAcceptanceCriteria:
    """Verify all acceptance criteria from the issue are met."""
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_all_acceptance_criteria(self, mock_openai_class):
        """Comprehensive test of all acceptance criteria from the issue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            project_info = pm.create_project("acceptance_test")
            
            llm = LLMInterface(project_name="acceptance_test")
            llm.persistence = pm
            
            # Mock successful response
            mock_response = AssistantResponse(
                explanation="I will analyze the data and create visualizations.",
                code="import pandas as pd\nimport matplotlib.pyplot as plt\n# Analysis code",
                assumptions=["Data is clean"],
                visualization_type="line_plot"
            )
            
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = "Mock LLM response"
            mock_completion.choices[0].message.parsed = mock_response
            mock_completion.usage.total_tokens = 500
            
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_completion
            mock_openai_class.return_value = mock_client
            
            llm._client = None
            llm.config.api_key = 'test-key'
            
            # Test with 3 different prompts and datasets (acceptance criteria requirement)
            test_cases = [
                {
                    'prompt': 'Plot revenue by month for the top 3 regions',
                    'context': {
                        'dataset_name': 'sales.csv',
                        'column_info': 'date, revenue, region',
                        'conventions': 'Use matplotlib'
                    }
                },
                {
                    'prompt': 'Analyze customer satisfaction trends',
                    'context': {
                        'dataset_name': 'customer_survey.csv',
                        'column_info': 'date, satisfaction, demographics',
                        'conventions': 'Use seaborn'
                    }
                },
                {
                    'prompt': 'Create financial performance dashboard',
                    'context': {
                        'dataset_name': 'financials.csv',
                        'column_info': 'quarter, metrics, performance',
                        'conventions': 'Use plotly'
                    }
                }
            ]
            
            responses = []
            for test_case in test_cases:
                response = llm.get_structured_response(
                    test_case['prompt'],
                    context=test_case['context']
                )
                responses.append(response)
            
            # Acceptance Criteria Verification:
            
            # 1. "Can send a prompt, receive a plan + code, and parse correctly"
            assert len(responses) == 3
            for response in responses:
                assert isinstance(response, AssistantResponse)
                assert len(response.explanation) > 0  # Plan
                assert len(response.code) > 0  # Code
                assert 'import' in response.code  # Executable Python
            
            # 2. "Full log entry appears in llm_log_*.jsonl"
            project_path = Path(tmpdir) / "acceptance_test"
            log_files = list(project_path.glob("llm_log_*.jsonl"))
            assert len(log_files) >= 1
            
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 3  # One entry per prompt
            
            for line in lines:
                log_entry = json.loads(line)
                # Verify exact format from issue
                required_fields = ['timestamp', 'user_prompt', 'llm_raw_response', 'llm_plan', 'llm_code']
                for field in required_fields:
                    assert field in log_entry
            
            # 3. "No hardcoded model parameters—use config or constants"
            config = llm.config
            assert hasattr(config, 'default_model')
            assert hasattr(config, 'default_temperature')
            assert hasattr(config, 'default_max_tokens')
            
            # 4. "Works with at least 3 example prompts and datasets"
            # Already verified above with 3 different test cases
            
            print("✅ All acceptance criteria verified successfully!")