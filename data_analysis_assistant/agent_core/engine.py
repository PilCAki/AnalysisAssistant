"""
Analysis engine - Core analysis planner and LLM prompt controller.

This module handles the main analysis logic, coordinating between
user requests, LLM interactions, and code generation.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type
from pydantic import BaseModel

import openai
from openai import OpenAI

from .models import AssistantResponse, LLMConfig, LLMLogEntry
from .persistence import PersistenceManager


class LLMInterface:
    """
    Interface for LLM interactions with structured output and comprehensive logging.
    
    Handles OpenAI API calls with Pydantic response formatting, error handling,
    retries, and detailed logging to JSONL files.
    """
    
    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize LLM interface.
        
        Args:
            project_name: Name of the project for logging (optional)
        """
        self.config = LLMConfig()
        self.project_name = project_name
        self.persistence = PersistenceManager()
        self._client = None
    
    @property
    def client(self) -> Optional[OpenAI]:
        """Get OpenAI client instance, creating it lazily."""
        if not self.config.is_configured:
            return None
        
        if self._client is None:
            self._client = OpenAI(api_key=self.config.api_key)
        
        return self._client
    
    def format_prompt(
        self, 
        user_request: str, 
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format conversation messages for OpenAI API.
        
        Args:
            user_request: User's analysis request
            context: Optional context including dataset info, conventions
            system_prompt: Optional custom system prompt
            
        Returns:
            Formatted message list for OpenAI API
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Load default system prompt from file
            prompt_file = Path(__file__).parent / "prompts" / "base_analysis_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    default_prompt = f.read()
                
                # Format with context if available
                if context:
                    formatted_prompt = default_prompt.format(
                        dataset_name=context.get('dataset_name', 'Unknown'),
                        column_info=context.get('column_info', 'Not specified'),
                        data_shape=context.get('data_shape', 'Not specified'),
                        data_types=context.get('data_types', 'Not specified'),
                        conventions=context.get('conventions', 'No specific conventions'),
                        user_request=user_request
                    )
                    messages.append({"role": "system", "content": formatted_prompt})
            
            if not messages:
                # Fallback system prompt
                default_system = (
                    "You are an expert data analyst. Generate Python code and explanations "
                    "for data analysis tasks. Respond with structured output including "
                    "explanation, code, and any assumptions made."
                )
                messages.append({"role": "system", "content": default_system})
        
        # Add user message
        if context and not system_prompt:
            # User message was already included in formatted system prompt
            pass
        else:
            messages.append({"role": "user", "content": user_request})
        
        return messages
    
    def call_llm(
        self,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel] = AssistantResponse,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> BaseModel:
        """
        Make a structured LLM API call with retries and logging.
        
        Args:
            messages: Conversation history for the API call
            response_format: Pydantic model class for structured output
            model: Model name (defaults to config default)
            temperature: Temperature setting (defaults to config default)
            max_tokens: Max tokens (defaults to config default)
            **kwargs: Additional arguments for OpenAI API
            
        Returns:
            Parsed response object of the specified response_format type
            
        Raises:
            Exception: If OpenAI client not available or API call fails after retries
        """
        if self.client is None:
            raise Exception(
                "OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
            )
        
        # Use defaults from config
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens
        
        # Generate unique prompt ID
        prompt_id = str(uuid.uuid4())
        
        # Prepare log entry
        log_entry = LLMLogEntry(
            prompt_id=prompt_id,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=messages[0]["content"] if messages and messages[0]["role"] == "system" else None,
            user_prompt=messages[-1]["content"] if messages else "",
            response_text="",  # Will be filled after successful call
        )
        
        start_time = time.time()
        last_exception = None
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    **kwargs
                )
                
                # Success - extract response and log
                parsed_response = completion.choices[0].message.parsed
                duration = time.time() - start_time
                
                log_entry.response_text = completion.choices[0].message.content or ""
                log_entry.structured_response = parsed_response.model_dump() if parsed_response else None
                log_entry.tokens_used = completion.usage.total_tokens if completion.usage else None
                log_entry.duration_seconds = duration
                
                # Log the interaction
                if self.project_name:
                    try:
                        self.persistence.log_llm_interaction(
                            self.project_name, 
                            log_entry.model_dump()
                        )
                    except Exception as e:
                        # Don't fail the LLM call if logging fails
                        print(f"Warning: Failed to log LLM interaction: {e}")
                
                return parsed_response
                
            except openai.RateLimitError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff for rate limits
                    wait_time = (2 ** attempt) * 1
                    time.sleep(wait_time)
                    continue
                
            except openai.APIError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)  # Brief wait before retry
                    continue
                
            except Exception as e:
                last_exception = e
                break  # Don't retry for unexpected errors
        
        # All retries failed - log error and raise
        duration = time.time() - start_time
        log_entry.error = str(last_exception)
        log_entry.duration_seconds = duration
        
        if self.project_name:
            try:
                self.persistence.log_llm_interaction(
                    self.project_name, 
                    log_entry.model_dump()
                )
            except Exception:
                pass  # Don't fail on logging error
        
        raise Exception(f"LLM API call failed after {self.config.max_retries} attempts: {last_exception}")
    
    def get_structured_response(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        response_format: Type[BaseModel] = AssistantResponse,
        **kwargs
    ) -> BaseModel:
        """
        High-level method to get structured response from LLM.
        
        Args:
            user_request: User's analysis request
            context: Optional context (dataset info, conventions, etc.)
            system_prompt: Optional custom system prompt
            response_format: Pydantic model for response structure
            **kwargs: Additional arguments for LLM call
            
        Returns:
            Structured response object
        """
        messages = self.format_prompt(user_request, context, system_prompt)
        return self.call_llm(messages, response_format, **kwargs)


class AnalysisEngine:
    """
    Main analysis engine that coordinates LLM interactions and analysis planning.
    
    Integrates the LLM interface with project management for comprehensive
    data analysis workflow orchestration.
    """
    
    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize analysis engine.
        
        Args:
            project_name: Name of the project for logging and context
        """
        self.project_name = project_name
        self.llm = LLMInterface(project_name)
        self.persistence = PersistenceManager()
    
    def plan_analysis(
        self, 
        user_request: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AssistantResponse:
        """
        Plan an analysis based on user request and available context.
        
        Args:
            user_request: User's analysis request
            context: Optional context including dataset info, project conventions
            
        Returns:
            Structured response with analysis plan and code
        """
        # Enhance context with project information if available
        if self.project_name and not context:
            try:
                project_info = self.persistence.get_project_info(self.project_name)
                context = {
                    'project_name': self.project_name,
                    'project_info': project_info
                }
            except Exception:
                context = {}
        
        return self.llm.get_structured_response(
            user_request=user_request,
            context=context,
            response_format=AssistantResponse
        )
    
    def generate_code(self, plan: Dict[str, Any]) -> str:
        """
        Generate executable code based on analysis plan.
        
        Args:
            plan: Analysis plan dictionary
            
        Returns:
            Generated Python code
        """
        if isinstance(plan, dict) and 'code' in plan:
            return plan['code']
        elif hasattr(plan, 'code'):
            return plan.code
        else:
            # If plan doesn't contain code, generate it
            request = f"Generate Python code for: {str(plan)}"
            response = self.llm.get_structured_response(request)
            return response.code