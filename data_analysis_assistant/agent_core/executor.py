"""
Code executor - Safe, state-aware code execution and traceback handler.

This module provides secure execution of generated analysis code
with proper error handling and state management.
"""

import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """
    Structured result object for code execution.

    Contains all information about code execution including success status,
    outputs, errors, and metadata.
    """

    success: bool = Field(description="Whether code executed successfully")
    stdout: str = Field(default="", description="Standard output from execution")
    stderr: str = Field(default="", description="Standard error from execution")
    exception: Optional[Dict[str, str]] = Field(
        default=None, description="Exception details if execution failed"
    )
    result: Optional[Any] = Field(
        default=None, description="Return value if code evaluation produced one"
    )
    code: str = Field(description="The code that was executed")
    execution_time: float = Field(description="Execution time in seconds")


def run_code(code: str, globals: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    """
    Execute Python code safely with comprehensive output and error capture.

    Args:
        code: Python code string to execute
        globals: Optional dictionary of global variables to inject

    Returns:
        ExecutionResult object with execution details
    """
    # Prepare execution namespace
    execution_globals = globals.copy() if globals else {}
    execution_globals.update(
        {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }
    )

    # Prepare output capture
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Initialize result
    result = ExecutionResult(success=False, code=code, execution_time=0.0)

    start_time = time.time()

    try:
        # Execute code with output redirection
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, execution_globals)

        # If we get here, execution was successful
        result.success = True
        result.stdout = stdout_capture.getvalue()
        result.stderr = stderr_capture.getvalue()

        # Save successful code to history
        _save_to_history(code)

    except Exception as e:
        # Capture exception details
        result.success = False
        result.stdout = stdout_capture.getvalue()
        result.stderr = stderr_capture.getvalue()

        # Format exception information
        full_traceback = traceback.format_exc()
        exception_type = type(e).__name__
        exception_message = str(e)

        # Create summary (type + first line of message)
        first_line = exception_message.split("\n")[0] if exception_message else ""
        summary = f"{exception_type}: {first_line}" if first_line else exception_type

        result.exception = {
            "type": exception_type,
            "message": exception_message,
            "summary": summary,
            "traceback": full_traceback,
        }

    finally:
        result.execution_time = time.time() - start_time

    return result


def _save_to_history(code: str) -> None:
    """
    Save successfully executed code to the history directory.

    Args:
        code: The code string to save
    """
    try:
        # Create history directory if it doesn't exist
        history_dir = Path("history")
        history_dir.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"code_{timestamp}.py"
        filepath = history_dir / filename

        # Ensure unique filename
        counter = 1
        while filepath.exists():
            filename = f"code_{timestamp}_{counter}.py"
            filepath = history_dir / filename
            counter += 1

        # Write code to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Code executed at {datetime.now().isoformat()}\n\n")
            f.write(code)

    except Exception:
        # Silently fail if history saving fails - don't break main execution
        pass


class CodeExecutor:
    """
    Handles safe execution of analysis code with error handling.

    This class provides backward compatibility and maintains state
    across multiple executions.
    """

    def __init__(self):
        self.execution_context = {}
        self.execution_history = []

    def execute_code(
        self, code: str, context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute Python code safely and return structured results.

        Args:
            code: Python code to execute
            context: Additional context variables

        Returns:
            ExecutionResult object with execution details
        """
        # Merge persistent context with provided context
        globals_dict = self.execution_context.copy()
        if context:
            globals_dict.update(context)

        # Execute code with modified run_code that returns globals
        result = self._run_code_with_context(code, globals_dict)

        # Update persistent context with any new variables if execution was successful
        if result.success:
            # Update context with new variables (excluding built-ins)
            for key, value in globals_dict.items():
                if not key.startswith("__"):
                    self.execution_context[key] = value
            self.execution_history.append(result)

        return result

    def _run_code_with_context(
        self, code: str, globals_dict: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute Python code with context capture.

        This is a modified version of run_code that allows capturing
        the updated globals for persistent context.
        """
        # Prepare execution namespace
        execution_globals = globals_dict.copy() if globals_dict else {}
        execution_globals.update(
            {
                "__builtins__": __builtins__,
                "__name__": "__main__",
            }
        )

        # Prepare output capture
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Initialize result
        result = ExecutionResult(success=False, code=code, execution_time=0.0)

        start_time = time.time()

        try:
            # Execute code with output redirection
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, execution_globals)

            # If we get here, execution was successful
            result.success = True
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            # Update the original globals_dict with new variables
            globals_dict.update(execution_globals)

            # Save successful code to history
            _save_to_history(code)

        except Exception as e:
            # Capture exception details
            result.success = False
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            # Format exception information
            full_traceback = traceback.format_exc()
            exception_type = type(e).__name__
            exception_message = str(e)

            # Create summary (type + first line of message)
            first_line = exception_message.split("\n")[0] if exception_message else ""
            summary = (
                f"{exception_type}: {first_line}" if first_line else exception_type
            )

            result.exception = {
                "type": exception_type,
                "message": exception_message,
                "summary": summary,
                "traceback": full_traceback,
            }

        finally:
            result.execution_time = time.time() - start_time

        return result
