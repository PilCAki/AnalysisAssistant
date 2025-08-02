"""
Pydantic models for structured LLM responses.

This module defines the data models used for structured communication
with LLM APIs, ensuring consistent response formats.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    """
    Standard structured response format for LLM interactions.
    
    This model ensures all LLM responses follow a consistent structure
    with explanation, code, and metadata fields.
    """
    explanation: str = Field(
        description="Clear explanation of the analysis approach and reasoning"
    )
    code: str = Field(
        description="Python code to execute for the analysis"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Any assumptions made about the data or analysis"
    )
    next_steps: Optional[str] = Field(
        default=None,
        description="Suggested next steps or follow-up questions"
    )
    visualization_type: Optional[str] = Field(
        default=None,
        description="Type of visualization being created (if applicable)"
    )


class LLMConfig:
    """
    Configuration for LLM API interactions.
    
    Handles environment variable loading and default settings.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model = "gpt-4o-mini"
        self.default_temperature = 0.7
        self.default_max_tokens = 1000
        self.timeout = 30
        self.max_retries = 3
    
    @property
    def is_configured(self) -> bool:
        """Check if OpenAI API key is available."""
        return self.api_key is not None and len(self.api_key.strip()) > 0


class LLMLogEntry(BaseModel):
    """
    Structure for logging LLM interactions to JSONL files.
    
    Contains all metadata needed for tracking and debugging LLM calls.
    """
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    prompt_id: str = Field(description="Unique identifier for this prompt")
    model: str = Field(description="LLM model used")
    temperature: float = Field(description="Temperature setting")
    max_tokens: int = Field(description="Maximum tokens setting")
    system_prompt: Optional[str] = Field(default=None, description="System prompt used")
    user_prompt: str = Field(description="User prompt sent")
    response_text: str = Field(description="Raw response from LLM")
    structured_response: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Parsed structured response"
    )
    tokens_used: Optional[int] = Field(default=None, description="Total tokens consumed")
    error: Optional[str] = Field(default=None, description="Error message if call failed")
    duration_seconds: Optional[float] = Field(default=None, description="Call duration")