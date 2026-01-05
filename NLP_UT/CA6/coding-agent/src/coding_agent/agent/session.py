"""Interactive session management for multi-turn conversations with bonus features."""

import uuid
import json
import pickle
import asyncio
from typing import Optional, List
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langchain_core.messages import HumanMessage, ToolMessage

from .tools import set_project_root, SENSITIVE_TOOLS
from .backend import build_agent_graph, usage_tracker

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

console = Console()


class InteractiveSession:
    def __init__(self, project_root: str):
        self.project_root = project_root
        set_project_root(project_root)

        # Initialize Checkpointer for Save/Load
        self.checkpointer = MemorySaver() if MemorySaver is not None else None
        self.graph = build_agent_graph(self.checkpointer)
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    async def process_turn(self, user_message: str) -> str:
        """Process turn with HITL, Pruning, and Usage Tracking."""

        # --- Handle Save Command ---
        if user_message.strip() == "/save":
            return self.save_session()

        # --- Handle Load Command ---
        if user_message.startswith("/load"):
            parts = user_message.split()
            filename = parts[1] if len(parts) > 1 else "session.pkl"
            return self.load_session(filename)

        inputs = {"messages": [HumanMessage(content=user_message)]}
        final_response = ""
        current_inputs = inputs
        resume_signal = None

        while True:
            try:
                events = self.graph.stream(
                    current_inputs if resume_signal is None else None,
                    self.config,
                    stream_mode="values",
                )

                for event in events:
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if getattr(last_msg, "type", None) == "ai" and getattr(
                            last_msg, "content", None
                        ):
                            final_response = last_msg.content

                snapshot = self.graph.get_state(self.config)
                if not snapshot.next:
                    break

                if "tools" in snapshot.next:
                    last_message = snapshot.values.get("messages", [])[-1]
                    tool_calls = getattr(last_message, "tool_calls", []) or []
                    if not tool_calls:
                        break

                    resume_signal = await self._handle_tool_approval(tool_calls)
                    if resume_signal == "denied":
                        current_inputs = None

            except Exception as e:
                return f"Error: {str(e)}"

        # Smart Context (Pruning)
        self._prune_context()

        # Append Usage Stats to Response
        stats = (
            f"\n\n[dim]Usage: {getattr(usage_tracker, 'total_tokens', 0)} tokens "
            f"(${getattr(usage_tracker, 'cost_est', 0.0):.4f})[/dim]"
        )
        return final_response + stats

    async def _handle_tool_approval(self, tool_calls) -> str:
        """Handle user approval for sensitive tools."""
        for tc in tool_calls:
            if tc.get("name") in SENSITIVE_TOOLS:
                console.print(
                    Panel(
                        f"[bold yellow]Tool:[/bold yellow] {tc.get('name')}\n[bold]Args:[/bold] {tc.get('args')}",
                        title="Permission Required",
                        border_style="yellow",
                    )
                )
                allow = await self._get_user_confirmation()
                if not allow:
                    denied_msg = ToolMessage(
                        tool_call_id=tc.get("id"),
                        content=f"Error: User denied {tc.get('name')}.",
                    )
                    # Update state to simulate denial
                    try:
                        self.graph.update_state(
                            self.config, {"messages": [denied_msg]}, as_node="tools"
                        )
                    except Exception:
                        # best-effort; continue
                        pass
                    return "denied"
        return "approved"

    async def _get_user_confirmation(self) -> bool:
        res = await asyncio.to_thread(
            Prompt.ask, "Allow?", choices=["y", "n"], default="n"
        )
        return res == "y"

    def _prune_context(self):
        """Summarize/Remove large file contents from history."""
        try:
            state = self.graph.get_state(self.config)
            messages = state.values.get("messages", [])
        except Exception:
            return

        pruned_messages = []
        modified = False

        for msg in messages:
            if (
                isinstance(msg, ToolMessage)
                and isinstance(getattr(msg, "content", None), str)
                and len(msg.content) > 2000
            ):
                new_content = (
                    msg.content[:1000]
                    + "\n...[CONTENT PRUNED]...\n"
                    + msg.content[-1000:]
                )
                msg.content = new_content
                modified = True
            pruned_messages.append(msg)

        if modified:
            try:
                self.graph.update_state(self.config, {"messages": pruned_messages})
            except Exception:
                pass

    def save_session(self, filename="session.pkl") -> str:
        """Save session state to file."""
        try:
            if self.checkpointer is None:
                return "Save not available: no checkpointer."

            with open(filename, "wb") as f:
                pickle.dump(getattr(self.checkpointer, "storage", None), f)

            return f"Session saved to {filename}"
        except Exception as e:
            return f"Error saving session: {str(e)}"

    def load_session(self, filename="session.pkl") -> str:
        """Load session state from file."""
        try:
            if not Path(filename).exists():
                return "File not found."

            if self.checkpointer is None:
                return "Load not available: no checkpointer."

            with open(filename, "rb") as f:
                storage = pickle.load(f)

            try:
                self.checkpointer.storage = storage
            except Exception:
                # best-effort assignment
                setattr(self.checkpointer, "storage", storage)

            return "Session loaded successfully."
        except Exception as e:
            return f"Error loading session: {str(e)}"
