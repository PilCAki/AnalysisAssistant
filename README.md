# AnalysisAssistant

A local LLM-powered data analysis assistant that provides an interactive chat interface for exploring, analyzing, and visualizing datasets. This tool combines the power of AI with code generation to help users perform data analysis tasks through natural language conversations.

## Features

- **Interactive Chat Interface**: Chat-based UI using Streamlit for natural data analysis conversations
- **File Upload**: Drag-and-drop support for datasets (CSV, XLSX, etc.)
- **AI-Powered Code Generation**: Automatic Python code generation for analysis tasks
- **Code Execution**: Safe execution of generated analysis code with error handling
- **Visualization**: Automatic plot generation using matplotlib, seaborn, and plotly
- **Project Persistence**: Save analysis history, conventions, and outputs per project
- **Modular Architecture**: Extensible design supporting future web deployment

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PilCAki/AnalysisAssistant.git
   cd AnalysisAssistant
   ```

2. **Install the package** (development mode):
   ```bash
   pip install -e .
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run data_analysis_assistant/ui/app_streamlit.py
   ```

   Or use the CLI command:
   ```bash
   data-assist --launch
   ```

## Project Structure

```
data_analysis_assistant/
├── agent_core/          # Core analysis engine and LLM integration
│   ├── engine.py        # Main analysis planner and LLM controller
│   ├── executor.py      # Code execution and error handling
│   ├── conversation.py  # Chat flow and memory management
│   ├── conventions.py   # Project conventions and configuration
│   ├── persistence.py   # State management and file I/O
│   ├── summarizer.py    # Analysis summarization
│   ├── tools/           # Analysis tools and utilities
│   └── prompts/         # LLM prompt templates
├── ui/                  # User interface components
│   ├── app_streamlit.py # Main Streamlit application
│   ├── components.py    # UI components (chat, plots, etc.)
│   └── state.py         # UI state management
└── projects/            # User project workspaces
```

## Usage

Once the Streamlit app is running:

1. **Upload a dataset** by dragging and dropping a CSV or Excel file
2. **Start a conversation** by describing what you want to analyze
3. **Review generated code** and results in the interface
4. **Iterate and refine** your analysis through follow-up questions
5. **Save your work** - all analysis scripts and outputs are automatically preserved

Example conversation starters:
- "Show me a summary of this dataset"
- "Plot the distribution of sales by region"
- "Find correlations between price and customer satisfaction"
- "Create a time series analysis of monthly revenue"

## Development

For development setup:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest

# Code formatting
black data_analysis_assistant/

# Type checking
mypy data_analysis_assistant/
```

## Configuration

The assistant uses OpenAI's API for LLM functionality. Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Project Architecture

This project is designed with modularity and extensibility in mind. See [plan.md](plan.md) for detailed architecture documentation and future roadmap.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.