"""Interactive session management for multi-turn conversations."""

import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

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

        self.agent = "Your agent instance here"  # TODO: Implement and initialize your agent

    async def process_turn(self, user_message: str) -> dict:
        """Process a single conversation turn.

        Args:
            user_message: User's input message

        Returns:
            Dictionary with turn results
        """
        # TODO: Implement agent processing logic
        return "Your agent response"
