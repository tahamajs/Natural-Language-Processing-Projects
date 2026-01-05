"""Builds the LangGraph agent graph used by the interactive session."""

import os
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except (
    Exception
):  # pragma: no cover - imported name fallback for environments without the package
    ChatOpenAI = None

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    BaseCallbackHandler = object

try:
    from langgraph.prebuilt import create_react_agent
except Exception:  # pragma: no cover
    create_react_agent = None

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover
    MemorySaver = None

from .tools import ALL_TOOLS

# Load environment variables (API Keys)
load_dotenv()


# --- Bonus: Usage Tracker ---
class TokenUsageTracker(BaseCallbackHandler):  # type: ignore[misc]
    """Tracks token usage for the session."""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_est = 0.0  # Estimate based on model rates

    def on_llm_end(self, response, **_kwargs):
        llm_output = getattr(response, "llm_output", None) or {}
        usage = (
            llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
        )
        if usage:
            self.total_tokens += usage.get("total_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)

            # Simple cost estimation (adjust rates as needed)
            input_cost = (self.prompt_tokens / 1_000_000) * 0.15
            output_cost = (self.completion_tokens / 1_000_000) * 0.60
            self.cost_est = input_cost + output_cost


# Global tracker instance
usage_tracker = TokenUsageTracker()


def build_agent_graph(checkpointer=None):
    """Builds and returns a compiled LangGraph agent (ReAct style)."""

    # Initialize the LLM
    llm = None
    if ChatOpenAI is not None:
        model_name = os.getenv("CODING_AGENT_MODEL_NAME", "gpt-4o-mini")
        temperature = float(os.getenv("CODING_AGENT_TEMPERATURE", "0.1"))
        # Attach the usage tracker callback if the LLM supports callbacks
        try:
            llm = ChatOpenAI(
                model=model_name, temperature=temperature, callbacks=[usage_tracker]
            )
        except Exception:
            llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Memory for the graph state (short-term session memory)
    if checkpointer is None and MemorySaver is not None:
        checkpointer = MemorySaver()

    # Provide a stronger system prompt / state modifier for the agent to enforce agency
    system_message = (
        "You are a professional coding agent.\n"
        "RULES:\n"
        "1. Before changing any code, CREATE or UPDATE a file named 'PLAN.md' "
        "   outlining your step-by-step approach.\n"
        "2. Always check directory structure first.\n"
        "3. Use 'grep_code' to find function definitions.\n"
        "4. If tests fail, update 'PLAN.md' with your hypothesis before trying again."
    )

    if create_react_agent is None:
        # Fallback: if LangGraph isn't installed, raise a clear error
        raise RuntimeError(
            "LangGraph 'create_react_agent' is not available in this environment."
        )
    # Note: different LangGraph versions accept different parameters.
    # Avoid passing `state_modifier` which may not be supported; rely on LLM/system-level prompts if needed.
    graph = create_react_agent(
        llm,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        # We interrupt before tools so the session can do HITL checks
        interrupt_before=["tools"],
    )

    return graph
