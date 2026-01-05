"""Interactive session management for multi-turn conversations."""

import uuid
import asyncio
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from langchain_core.messages import HumanMessage

from .tools import set_project_root, SENSITIVE_TOOLS
from .backend import build_agent_graph

console = Console()


class InteractiveSession:
    """Manages an interactive multi-turn coding session."""

    def __init__(
        self,
        project_root: str,
    ):
        """Initialize interactive session.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root

        # Initialize tools context
        set_project_root(project_root)

        # Initialize the Graph
        self.graph = build_agent_graph()

        # Create a unique thread ID for memory management
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def process_turn(self, user_message: str) -> str:
        """Process a single conversation turn with Human-in-the-Loop logic.

        Args:
            user_message: User's input message

        Returns:
            The final textual response from the agent.
        """

        # 1. Add user message to the state
        inputs = {"messages": [HumanMessage(content=user_message)]}

        final_response = ""

        # Start the graph execution
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
                    # Capture the latest AI message as the potential response
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and last_msg.content:
                            final_response = last_msg.content

                # Check the status of the graph
                snapshot = self.graph.get_state(self.config)

                if not snapshot.next:
                    # Graph execution finished successfully
                    break

                if "tools" in snapshot.next:
                    # The graph stopped before executing tools (HITL)
                    # We need to inspect what tool it wants to run
                    last_message = snapshot.values.get("messages", [])[-1]

                    tool_calls = getattr(last_message, "tool_calls", []) or []

                    for tc in tool_calls:
                        tool_name = tc.get("name")
                        tool_args = tc.get("args")

                        # Check if sensitive
                        if tool_name in SENSITIVE_TOOLS:
                            console.print(
                                Panel(
                                    f"[bold yellow]Tool Request:[/bold yellow] {tool_name}\n"
                                    f"[bold]Args:[/bold] {tool_args}",
                                    title="Permission Required",
                                    border_style="yellow",
                                )
                            )

                            answer = await self._get_user_confirmation()

                            if not answer:
                                # User denied. We update state to simulate a tool error
                                from langchain_core.messages import ToolMessage

                                denied_msg = ToolMessage(
                                    tool_call_id=tc.get("id"),
                                    content=f"Error: User denied permission to execute {tool_name}.",
                                )
                                # Update the graph state with the denial so the agent can react
                                self.graph.update_state(self.config, {"messages": [denied_msg]}, as_node="tools")

                                resume_signal = "denied"
                            else:
                                resume_signal = "approved"
                        else:
                            resume_signal = "auto"

                    # After handling approvals, resume from the updated state
                    current_inputs = None

            except Exception as e:
                return f"An internal error occurred: {str(e)}"

        return final_response

    async def _get_user_confirmation(self) -> bool:
        """Ask user for confirmation via Rich prompt."""
        response = await asyncio.to_thread(Prompt.ask, "[bold yellow]Allow execution?[/bold yellow]", choices=["y", "n"], default="n")
        return response == "y"
