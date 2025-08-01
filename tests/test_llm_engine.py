"""
Tests for LLM engine functionality.

Tests the LLM interface, structured responses, logging, and error handling.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from data_analysis_assistant.agent_core.engine import LLMInterface, AnalysisEngine
from data_analysis_assistant.agent_core.models import AssistantResponse, LLMConfig, LLMLogEntry
from data_analysis_assistant.agent_core.summary_generator import SessionSummarizer


class TestLLMConfig:
    """Test LLM configuration management."""
    
    def test_config_initialization(self):
        """Test basic config initialization."""
        config = LLMConfig()
        assert config.default_model == "gpt-4o-mini"
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 1000
        assert config.max_retries == 3
    
    def test_config_with_api_key(self):
        """Test config when API key is set."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
            config = LLMConfig()
            assert config.is_configured is True
            assert config.api_key == 'test-key-123'
    
    def test_config_without_api_key(self):
        """Test config when API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.is_configured is False


class TestAssistantResponse:
    """Test structured response models."""
    
    def test_assistant_response_creation(self):
        """Test creating AssistantResponse with required fields."""
        response = AssistantResponse(
            explanation="This is a test analysis",
            code="print('hello world')"
        )
        assert response.explanation == "This is a test analysis"
        assert response.code == "print('hello world')"
        assert response.assumptions == []
        assert response.next_steps is None
    
    def test_assistant_response_with_all_fields(self):
        """Test creating AssistantResponse with all fields."""
        response = AssistantResponse(
            explanation="Analysis explanation",
            code="import pandas as pd",
            assumptions=["Data is clean", "No missing values"],
            next_steps="Consider more features",
            visualization_type="scatter plot"
        )
        assert len(response.assumptions) == 2
        assert response.next_steps == "Consider more features"
        assert response.visualization_type == "scatter plot"


class TestLLMLogEntry:
    """Test LLM log entry model."""
    
    def test_log_entry_creation(self):
        """Test creating log entry with required fields."""
        entry = LLMLogEntry(
            prompt_id="test-123",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            user_prompt="Test prompt",
            response_text="Test response"
        )
        assert entry.prompt_id == "test-123"
        assert entry.model == "gpt-4o-mini"
        assert entry.temperature == 0.7
        assert entry.user_prompt == "Test prompt"
        assert entry.response_text == "Test response"
        assert entry.error is None


class TestLLMInterface:
    """Test LLM interface functionality."""
    
    def test_initialization_without_api_key(self):
        """Test LLM interface initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            llm = LLMInterface()
            assert llm.client is None
            assert not llm.config.is_configured
    
    def test_initialization_with_project(self):
        """Test LLM interface initialization with project name."""
        llm = LLMInterface(project_name="test_project")
        assert llm.project_name == "test_project"
    
    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        llm = LLMInterface()
        messages = llm.format_prompt("Analyze this data")
        
        assert len(messages) >= 1
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Analyze this data"
    
    def test_format_prompt_with_context(self):
        """Test prompt formatting with context."""
        llm = LLMInterface()
        context = {
            'dataset_name': 'sales_data.csv',
            'column_info': 'date, sales, region',
            'data_shape': '(1000, 3)',
            'data_types': 'datetime, float, str',
            'conventions': 'Use snake_case for variables'
        }
        
        messages = llm.format_prompt("Plot sales by region", context)
        
        # Should have system and user messages
        assert len(messages) >= 1
        # System message should contain context information
        system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""
        assert "sales_data.csv" in system_content
    
    def test_format_prompt_with_custom_system(self):
        """Test prompt formatting with custom system prompt."""
        llm = LLMInterface()
        custom_system = "You are a specialized data scientist."
        messages = llm.format_prompt("Analyze data", system_prompt=custom_system)
        
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_system
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Analyze data"
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    def test_call_llm_no_client(self, mock_openai):
        """Test LLM call when client is not available."""
        with patch.dict(os.environ, {}, clear=True):
            llm = LLMInterface()
            messages = [{"role": "user", "content": "test"}]
            
            with pytest.raises(Exception, match="OpenAI client not initialized"):
                llm.call_llm(messages)
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    def test_call_llm_success(self, mock_openai):
        """Test successful LLM call."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_usage = Mock()
        
        # Set up the mock chain
        mock_message.parsed = AssistantResponse(
            explanation="Test explanation",
            code="print('test')"
        )
        mock_message.content = "Test response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage.total_tokens = 100
        mock_response.usage = mock_usage
        
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            llm = LLMInterface()
            messages = [{"role": "user", "content": "test prompt"}]
            
            result = llm.call_llm(messages)
            
            assert isinstance(result, AssistantResponse)
            assert result.explanation == "Test explanation"
            assert result.code == "print('test')"
            
            # Verify the API was called correctly
            mock_client.beta.chat.completions.parse.assert_called_once()
            call_args = mock_client.beta.chat.completions.parse.call_args
            assert call_args[1]['model'] == 'gpt-4o-mini'
            assert call_args[1]['messages'] == messages
            assert call_args[1]['response_format'] == AssistantResponse
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    def test_get_structured_response(self, mock_openai):
        """Test high-level structured response method."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_usage = Mock()
        
        mock_message.parsed = AssistantResponse(
            explanation="Analysis explanation",
            code="df.plot()",
            visualization_type="line plot"
        )
        mock_message.content = "Response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            llm = LLMInterface()
            
            result = llm.get_structured_response(
                user_request="Create a plot of sales over time",
                context={'dataset_name': 'sales.csv'}
            )
            
            assert isinstance(result, AssistantResponse)
            assert result.explanation == "Analysis explanation"
            assert result.code == "df.plot()"
            assert result.visualization_type == "line plot"


class TestAnalysisEngine:
    """Test analysis engine functionality."""
    
    def test_initialization(self):
        """Test analysis engine initialization."""
        engine = AnalysisEngine()
        assert engine.project_name is None
        assert isinstance(engine.llm, LLMInterface)
    
    def test_initialization_with_project(self):
        """Test analysis engine initialization with project."""
        engine = AnalysisEngine(project_name="test_project")
        assert engine.project_name == "test_project"
        assert engine.llm.project_name == "test_project"
    
    @patch('data_analysis_assistant.agent_core.engine.OpenAI')
    def test_plan_analysis(self, mock_openai):
        """Test analysis planning."""
        # Mock the OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_usage = Mock()
        
        mock_message.parsed = AssistantResponse(
            explanation="Plan: Analyze sales trends",
            code="df.groupby('month').sum().plot()",
            assumptions=["Data is monthly aggregated"]
        )
        mock_message.content = "Response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage.total_tokens = 200
        mock_response.usage = mock_usage
        
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            engine = AnalysisEngine()
            
            result = engine.plan_analysis("Analyze monthly sales trends")
            
            assert isinstance(result, AssistantResponse)
            assert "sales trends" in result.explanation
            assert "plot" in result.code
            assert len(result.assumptions) == 1
    
    def test_generate_code_from_response(self):
        """Test code generation from AssistantResponse."""
        engine = AnalysisEngine()
        
        response = AssistantResponse(
            explanation="Test explanation",
            code="import pandas as pd\ndf.head()"
        )
        
        code = engine.generate_code(response)
        assert code == "import pandas as pd\ndf.head()"
    
    def test_generate_code_from_dict(self):
        """Test code generation from dictionary."""
        engine = AnalysisEngine()
        
        plan = {'code': 'print("hello world")'}
        code = engine.generate_code(plan)
        assert code == 'print("hello world")'


class TestSessionSummarizer:
    """Test session summarizer functionality."""
    
    def test_initialization(self):
        """Test summarizer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            summarizer = SessionSummarizer(temp_dir)
            assert summarizer.project_path == Path(temp_dir)
    
    def test_analyze_empty_session(self):
        """Test analyzing empty session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            summarizer = SessionSummarizer(temp_dir)
            analysis = summarizer.analyze_session([])
            assert analysis == {}
    
    def test_analyze_session_with_entries(self):
        """Test analyzing session with log entries."""
        entries = [
            {
                'timestamp': '2024-01-01T10:00:00',
                'model': 'gpt-4o-mini',
                'tokens_used': 100,
                'user_prompt': 'Plot sales data',
                'structured_response': {
                    'code': 'df.plot()',
                    'visualization_type': 'line plot',
                    'assumptions': ['Data is clean']
                }
            },
            {
                'timestamp': '2024-01-01T10:05:00',
                'model': 'gpt-4o-mini',
                'tokens_used': 150,
                'user_prompt': 'Show correlation matrix',
                'error': 'API timeout'
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            summarizer = SessionSummarizer(temp_dir)
            analysis = summarizer.analyze_session(entries)
            
            assert analysis['total_interactions'] == 2
            assert analysis['successful_calls'] == 1
            assert analysis['failed_calls'] == 1
            assert analysis['total_tokens'] == 250
            assert len(analysis['questions_asked']) == 2
            assert len(analysis['code_generated']) == 1
            assert len(analysis['errors_encountered']) == 1
    
    def test_generate_summary_markdown(self):
        """Test markdown summary generation."""
        analysis = {
            'total_interactions': 2,
            'session_start': '2024-01-01T10:00:00',
            'session_end': '2024-01-01T10:05:00',
            'models_used': ['gpt-4o-mini'],
            'total_tokens': 250,
            'successful_calls': 1,
            'failed_calls': 1,
            'questions_asked': [
                {'timestamp': '2024-01-01T10:00:00', 'question': 'Plot sales data'}
            ],
            'code_generated': [
                {'timestamp': '2024-01-01T10:00:00', 'code_snippet': 'df.plot()'}
            ],
            'visualizations_created': [],
            'assumptions_made': [],
            'errors_encountered': [],
            'main_topics': {'plot': 1}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            summarizer = SessionSummarizer(temp_dir)
            markdown = summarizer.generate_summary_markdown(analysis)
            
            assert "# LLM Session Summary" in markdown
            assert "Total Interactions:** 2" in markdown
            assert "gpt-4o-mini" in markdown
            assert "Plot sales data" in markdown
            assert "df.plot()" in markdown
    
    def test_create_session_summary_file(self):
        """Test creating summary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a mock log file
            log_file = temp_path / "llm_log_20240101_100000.jsonl"
            with open(log_file, 'w') as f:
                entry = {
                    'timestamp': '2024-01-01T10:00:00',
                    'model': 'gpt-4o-mini',
                    'user_prompt': 'Test prompt',
                    'response_text': 'Test response'
                }
                f.write(json.dumps(entry) + '\n')
            
            summarizer = SessionSummarizer(temp_dir)
            summary_path = summarizer.create_session_summary()
            
            assert Path(summary_path).exists()
            assert Path(summary_path).name == "llm_summary.md"
            
            # Verify content
            with open(summary_path, 'r') as f:
                content = f.read()
                assert "# LLM Session Summary" in content
                assert "Test prompt" in content


if __name__ == "__main__":
    pytest.main([__file__])