"""Filesystem and shell tools used by the coding agent."""
import os
import subprocess
from pathlib import Path
from typing import List, Optional

try:
    # langchain_core may not be installed in all environments; keep the import here
    from langchain_core.tools import tool
except Exception:  # pragma: no cover - fall back for environments without langchain_core
    def tool(func=None, **_kwargs):
        # simple decorator passthrough when the real tool decorator isn't available
        if func is None:
            return lambda f: f
        return func

# Global variable to store the project root, set during initialization
PROJECT_ROOT = Path.cwd()

def set_project_root(path: str):
    global PROJECT_ROOT
    PROJECT_ROOT = Path(path).resolve()

def _get_safe_path(file_path: str) -> Path:
    """Ensure the path is within the project root to prevent unauthorized access."""
    target = (PROJECT_ROOT / file_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Access denied: {file_path} is outside project root.")
    return target


@tool
def list_files(directory: str = ".") -> str:
    """List files and directories in the specified path relative to project root."""
    try:
        target_dir = _get_safe_path(directory)
        if not target_dir.exists():
            return f"Error: Directory {directory} does not exist."

        items = []
        for item in target_dir.iterdir():
            if item.name.startswith("."):  # Skip hidden files
                continue
            kind = "DIR" if item.is_dir() else "FILE"
            items.append(f"[{kind}] {item.name}")
        return "\n".join(sorted(items))
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """Read the content of a file."""
    try:
        target = _get_safe_path(file_path)
        if not target.exists():
            return f"Error: File {file_path} does not exist."
        if not target.is_file():
            return f"Error: {file_path} is not a file."

        return target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def create_file(file_path: str, content: str) -> str:
    """Create a new file with the given content. Fails if file already exists."""
    try:
        target = _get_safe_path(file_path)
        if target.exists():
            return f"Error: File {file_path} already exists. Use overwrite_file instead."

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully created {file_path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def overwrite_file(file_path: str, content: str) -> str:
    """Overwrite an existing file with new content."""
    try:
        target = _get_safe_path(file_path)
        if not target.exists():
            return f"Error: File {file_path} does not exist. Use create_file instead."

        target.write_text(content, encoding="utf-8")
        return f"Successfully overwritten {file_path}"
    except Exception as e:
        return f"Error overwriting file: {str(e)}"


@tool
def search_files(pattern: str) -> str:
    """Search for files matching a glob pattern (e.g., '*.py') recursively."""
    try:
        results = []
        for path in PROJECT_ROOT.rglob(pattern):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            results.append(str(path.relative_to(PROJECT_ROOT)))

        if not results:
            return "No files found matching the pattern."
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {str(e)}"


@tool
def execute_shell(command: str) -> str:
    """Execute a shell command. Use with caution."""
    # Security Note: In a real prod env, this needs strict sandboxing.
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30  # Timeout to prevent hanging
        )
        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output if output.strip() else "Command executed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return f"Error executing command: {str(e)}"


# List of tools exported for the agent
ALL_TOOLS = [list_files, read_file, create_file, overwrite_file, search_files, execute_shell]
# Tools that require Human-in-the-Loop confirmation
SENSITIVE_TOOLS = {"create_file", "overwrite_file", "execute_shell"}

