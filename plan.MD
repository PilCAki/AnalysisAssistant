📄 plan.md — Local LLM-Powered Data Analysis Assistant
🧠 Project Goal
Create a local desktop application for AI-powered, iterative data analysis. The assistant mimics (and surpasses) Microsoft’s Analyst Agent, allowing a user to drag-and-drop a dataset and interact with a helpful, chat-based interface that:

Understands the analysis context

Writes and revises code

Executes and visualizes results

Asks clarification questions when uncertain

Stores reusable, editable analysis scripts

Persists reasoning context across sessions

🏗️ System Architecture Overview
java
Copy
Edit
Local Desktop App (Streamlit or PySide6/Gradio)
     ├── LLM Analysis Agent (OpenAI API: GPT-4.1, 03, 04-mini-high)
     ├── Code Execution Sandbox (Local Python exec())
     ├── Project Workspace (dataset, script history, config, outputs)
     └── Convention + Context Manager (disk-persisted, JSON-based)
🗂️ Repository Structure
bash
Copy
Edit
data_analysis_assistant/
│
├── agent_core/
│   ├── engine.py              # Core analysis planner + LLM prompt controller
│   ├── executor.py            # Safe, state-aware code execution + traceback handler
│   ├── conversation.py        # Handles chat flow, memory, summaries
│   ├── conventions.py         # Convention inference, user prompts, and project config
│   ├── persistence.py         # Read/write project state, LLM logs, script files
│   ├── tools/
│   │   ├── profiling.py       # pandas-profiling wrapper
│   │   ├── plotting.py        # Plotly/matplotlib/seaborn integrations
│   │   └── modeling.py        # sklearn modeling helpers
│   └── prompts/
│       ├── base_analysis_prompt.txt
│       └── error_fixing_prompt.txt
│
├── ui/
│   ├── app_streamlit.py       # Main chat-style UI (Streamlit desktop app)
│   ├── components.py          # Chat box, code viewer, plot area
│   ├── state.py               # UI state manager and backend glue
│
├── projects/
│   └── my_first_project/
│       ├── data/
│       │   └── sales.csv
│       ├── history/
│       │   ├── 001_load_and_clean.py
│       │   ├── 002_plot_trends.py
│       ├── conventions.json
│       ├── llm_log_2025-08-01.jsonl
│       ├── llm_summary.md
│       └── outputs/
│           └── plots/
│               └── sales_trends.png
│
├── cli.py                     # Command-line entrypoint (future)
├── setup.py / pyproject.toml  # pip install support
├── requirements.txt
├── README.md
└── plan.md
🚀 Features
✅ Core Capabilities
Capability	Details
Chat Interface	Text-based chat UI using Streamlit or Gradio
File Upload	Drag-and-drop datasets (CSV, XLSX, etc.)
Prompt-Driven Analysis	"Plot revenue over time" → LLM → working code
Automatic Code Revision	Executes code and retries on failure
Clarification Loop	Asks user for missing info (e.g., column semantics)
Convention Memory	Infers and/or asks for project rules (e.g., time column format)
Code History	Persists scripts by step, editable and rerunnable
Output Viewer	Plots, tables, and logs shown in UI
Session Memory	LLM reasoning and prior Q&A saved in JSONL + Markdown
Profiling + Modeling Tools	Pandas profiling, plotting libs, sklearn pipelines

🛠️ Technologies
Component	Tool
UI	Streamlit (desktop first)
LLM API	OpenAI GPT-4.1 / 03 / 04-mini-high
Python	3.11+
Plotting	matplotlib, seaborn, plotly
Profiling	pandas-profiling, sweetviz (optional)
Modeling	scikit-learn
Persistence	JSON (for state), .py (for scripts), Markdown (for summaries)

📁 Project Folder Format
Each user project is stored in a folder:

bash
Copy
Edit
projects/my_project/
├── data/                    # Raw uploaded datasets
├── history/                 # Scripts generated step by step
├── conventions.json         # Known rules: date format, ID col, etc.
├── llm_log_*.jsonl          # Full LLM interaction logs with timestamps
├── llm_summary.md           # Human-readable recap of chat history
└── outputs/                 # Figures, tables, models, etc.
🔄 Chat/Execution Loop
mermaid
Copy
Edit
graph TD
A[User Prompt] --> B[LLM Generates Plan + Code]
B --> C[Executor Runs Code]
C -->|Success| D[Show Plot / Result]
C -->|Failure| E[LLM Revises Code]
E --> C
B --> F{Missing info?}
F -->|Yes| G[Ask User for Clarification]
G --> B
📦 Packaging and Distribution
setup.py + pyproject.toml for pip-installation

CLI and app launch via:

bash
Copy
Edit
data-assist --launch
Optional: system tray launcher script for Mac/Windows/Linux

🧪 Initial Milestones
Milestone	Description
✅ Basic file drop + chat UI in Streamlit	
✅ LLM prompt → code generation	
✅ Code execution + plot return	
✅ Convention prompt + memory (per-project)	
🟡 Script history manager (save, edit, rerun)	
🟡 LLM logging + summary	
🟡 Error fixing loop	
🔲 pandas profiling + sklearn helpers	
🔲 Final code export (pipeline + notebook)	

🔮 Future Extensions
Web deployment using FastAPI + React

Multi-agent plugin interface for automating entire analysis flows

Live notebook integration

Local model backend (Ollama, Mistral)

Voice interaction

LLM training data generation mode (e.g., for evaluation)
