"""
LLM session summary generator.

This module generates human-readable summaries of LLM interactions
from JSONL log files, creating comprehensive session reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


class SessionSummarizer:
    """
    Generates human-readable summaries from LLM interaction logs.
    
    Analyzes JSONL log files to create comprehensive summaries of
    analysis sessions, including questions asked, goals, and insights.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize summarizer for a specific project.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
    
    def load_log_entries(self, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load LLM log entries from JSONL files.
        
        Args:
            log_file: Specific log file to load (optional, loads most recent if None)
            
        Returns:
            List of log entry dictionaries
        """
        entries = []
        
        if log_file:
            log_path = self.project_path / log_file
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        else:
            # Load all log files
            for log_path in sorted(self.project_path.glob("llm_log_*.jsonl")):
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        
        return entries
    
    def analyze_session(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze log entries to extract session insights.
        
        Args:
            entries: List of log entry dictionaries
            
        Returns:
            Dictionary containing session analysis
        """
        if not entries:
            return {}
        
        analysis = {
            'total_interactions': len(entries),
            'session_start': entries[0].get('timestamp', 'Unknown'),
            'session_end': entries[-1].get('timestamp', 'Unknown'),
            'models_used': set(),
            'total_tokens': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'questions_asked': [],
            'code_generated': [],
            'visualizations_created': [],
            'assumptions_made': [],
            'errors_encountered': [],
            'main_topics': defaultdict(int)
        }
        
        for entry in entries:
            # Track models and tokens
            if 'model' in entry:
                analysis['models_used'].add(entry['model'])
            
            if 'tokens_used' in entry and entry['tokens_used']:
                analysis['total_tokens'] += entry['tokens_used']
            
            # Track success/failure
            if entry.get('error'):
                analysis['failed_calls'] += 1
                analysis['errors_encountered'].append({
                    'timestamp': entry.get('timestamp'),
                    'error': entry.get('error'),
                    'prompt': entry.get('user_prompt', '')[:100] + '...' if len(entry.get('user_prompt', '')) > 100 else entry.get('user_prompt', '')
                })
            else:
                analysis['successful_calls'] += 1
            
            # Extract user questions
            user_prompt = entry.get('user_prompt', '')
            if user_prompt:
                analysis['questions_asked'].append({
                    'timestamp': entry.get('timestamp'),
                    'question': user_prompt[:200] + '...' if len(user_prompt) > 200 else user_prompt
                })
            
            # Extract structured response data
            structured_response = entry.get('structured_response', {})
            if structured_response:
                if 'code' in structured_response and structured_response['code']:
                    analysis['code_generated'].append({
                        'timestamp': entry.get('timestamp'),
                        'code_snippet': structured_response['code'][:300] + '...' if len(structured_response['code']) > 300 else structured_response['code']
                    })
                
                if 'assumptions' in structured_response and structured_response['assumptions']:
                    analysis['assumptions_made'].extend(structured_response['assumptions'])
                
                if 'visualization_type' in structured_response and structured_response['visualization_type']:
                    analysis['visualizations_created'].append({
                        'timestamp': entry.get('timestamp'),
                        'type': structured_response['visualization_type']
                    })
            
            # Extract topics from prompts (simple keyword extraction)
            prompt_text = user_prompt.lower()
            keywords = ['plot', 'chart', 'graph', 'analyze', 'correlation', 'regression', 
                       'classification', 'clustering', 'visualization', 'summary', 'statistics']
            for keyword in keywords:
                if keyword in prompt_text:
                    analysis['main_topics'][keyword] += 1
        
        # Convert sets to lists for JSON serialization
        analysis['models_used'] = list(analysis['models_used'])
        analysis['main_topics'] = dict(analysis['main_topics'])
        
        return analysis
    
    def generate_summary_markdown(self, analysis: Dict[str, Any]) -> str:
        """
        Generate markdown summary from session analysis.
        
        Args:
            analysis: Session analysis dictionary
            
        Returns:
            Markdown formatted summary string
        """
        if not analysis:
            return "# LLM Session Summary\n\nNo interactions found.\n"
        
        md = []
        md.append("# LLM Session Summary")
        md.append("")
        md.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")
        
        # Session Overview
        md.append("## Session Overview")
        md.append("")
        md.append(f"- **Total Interactions:** {analysis['total_interactions']}")
        md.append(f"- **Session Duration:** {analysis['session_start']} to {analysis['session_end']}")
        md.append(f"- **Models Used:** {', '.join(analysis['models_used'])}")
        md.append(f"- **Total Tokens Consumed:** {analysis['total_tokens']:,}")
        md.append(f"- **Successful Calls:** {analysis['successful_calls']}")
        md.append(f"- **Failed Calls:** {analysis['failed_calls']}")
        md.append("")
        
        # Main Topics
        if analysis['main_topics']:
            md.append("## Main Topics Discussed")
            md.append("")
            for topic, count in sorted(analysis['main_topics'].items(), key=lambda x: x[1], reverse=True):
                md.append(f"- **{topic.title()}:** {count} mentions")
            md.append("")
        
        # Questions Asked
        if analysis['questions_asked']:
            md.append("## Questions Asked")
            md.append("")
            for i, q in enumerate(analysis['questions_asked'][:10], 1):  # Limit to first 10
                timestamp = q['timestamp'][:19] if q['timestamp'] else 'Unknown'
                md.append(f"{i}. *({timestamp})* {q['question']}")
            if len(analysis['questions_asked']) > 10:
                md.append(f"... and {len(analysis['questions_asked']) - 10} more questions")
            md.append("")
        
        # Code Generated
        if analysis['code_generated']:
            md.append("## Code Generated")
            md.append("")
            for i, code in enumerate(analysis['code_generated'][:5], 1):  # Limit to first 5
                timestamp = code['timestamp'][:19] if code['timestamp'] else 'Unknown'
                md.append(f"### Code Block {i} ({timestamp})")
                md.append("```python")
                md.append(code['code_snippet'])
                md.append("```")
                md.append("")
            if len(analysis['code_generated']) > 5:
                md.append(f"... and {len(analysis['code_generated']) - 5} more code blocks")
            md.append("")
        
        # Visualizations
        if analysis['visualizations_created']:
            md.append("## Visualizations Created")
            md.append("")
            viz_types = defaultdict(int)
            for viz in analysis['visualizations_created']:
                viz_types[viz['type']] += 1
            for viz_type, count in viz_types.items():
                md.append(f"- **{viz_type}:** {count} visualization(s)")
            md.append("")
        
        # Assumptions Made
        if analysis['assumptions_made']:
            md.append("## Key Assumptions")
            md.append("")
            unique_assumptions = list(set(analysis['assumptions_made']))[:10]  # Remove duplicates, limit to 10
            for assumption in unique_assumptions:
                md.append(f"- {assumption}")
            md.append("")
        
        # Errors
        if analysis['errors_encountered']:
            md.append("## Errors Encountered")
            md.append("")
            for error in analysis['errors_encountered'][:5]:  # Limit to first 5
                timestamp = error['timestamp'][:19] if error['timestamp'] else 'Unknown'
                md.append(f"- *({timestamp})* {error['error']}")
                md.append(f"  - Prompt: {error['prompt']}")
            if len(analysis['errors_encountered']) > 5:
                md.append(f"... and {len(analysis['errors_encountered']) - 5} more errors")
            md.append("")
        
        md.append("---")
        md.append("*This summary was automatically generated from LLM interaction logs.*")
        
        return "\n".join(md)
    
    def create_session_summary(self, output_file: str = "llm_summary.md") -> str:
        """
        Create a complete session summary and save to file.
        
        Args:
            output_file: Name of the output markdown file
            
        Returns:
            Path to the created summary file
        """
        entries = self.load_log_entries()
        analysis = self.analyze_session(entries)
        summary_md = self.generate_summary_markdown(analysis)
        
        output_path = self.project_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_md)
        
        return str(output_path)