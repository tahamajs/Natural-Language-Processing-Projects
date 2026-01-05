"""Interactive CLI mode for multi-turn conversations."""

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from coding_agent.agent.session import InteractiveSession

console = Console()


class InteractiveCLI:
    """Interactive command-line interface for the coding agent."""

    def __init__(self, session: InteractiveSession):
        """Initialize interactive CLI.

        Args:
            session: Interactive session instance
        """
        self.session = session
        self.running = True

    async def run(self) -> None:
        """Run the interactive session."""
        self._print_welcome()
        self._print_help()

        while self.running:
            try:
                # Get user input
                user_input = await self._get_input()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Process as a normal turn
                await self._process_turn(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to quit or continue typing[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                console.print("[yellow]Session continues...[/yellow]\n")

        self._print_goodbye()

    def _print_welcome(self) -> None:
        """Print welcome message."""
        console.print(Panel(
            "[bold cyan]Coding Agent[/bold cyan]\n\n"
            f"Project: {self.session.project_root}\n"
            "Type your coding tasks or questions. Use /help for commands.",
            border_style="cyan",
            title="Welcome",
        ))

    def _print_help(self) -> None:
        """Print help message."""
        help_text = """[bold]Available Commands:[/bold]

[cyan]/help[/cyan]        - Show this help message
[cyan]/save[/cyan]        - Save current session (Bonus)
[cyan]/load [file][/cyan] - Load session from file (Bonus)
[cyan]/exit[/cyan]        - Exit the session

[bold]Tips:[/bold]
- The agent remembers your conversation history
- Recently accessed files are tracked automatically
- Use natural language to describe tasks
"""
        console.print(Panel(help_text, border_style="blue", title="Help"))

    def _print_goodbye(self) -> None:
        """Print goodbye message."""
        
        console.print(Panel(
            "[bold cyan]Session Ended[/bold cyan]\n\n"
            "Thank you for using Coding Agent!",
            border_style="cyan",
            title="Goodbye",
        ))

    async def _get_input(self) -> str:
        """Get user input.

        Returns:
            User input string
        """
        # Get input
        user_input = await asyncio.to_thread(
            Prompt.ask,
            "\n[bold green]You[/bold green]",
        )

        return user_input.strip()

    async def _process_turn(self, user_message: str) -> None:
        """Process a conversation turn.

        Args:
            user_message: User's message
        """

        try:
            # Process the turn
            turn_result = await self.session.process_turn(user_message)

            # Display response
            console.print(f"\n[bold blue]Agent[/bold blue]:")
            console.print(Panel(
                turn_result,
                border_style="blue",
                title="Response",
            ))

        except Exception as e:
            console.print(f"[red]Error processing turn: {str(e)}[/red]")

    async def _handle_command(self, command: str) -> None:
        """Handle special commands.

        Args:
            command: Command string starting with /
        """
        cmd = command.lower().split()[0]

        if cmd == "/help":
            self._print_help()

        elif cmd == "/save":
            try:
                result = self.session.save_session()
                console.print(f"[green]{result}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving session: {e}[/red]")

        elif cmd.startswith("/load"):
            parts = command.split()
            filename = parts[1] if len(parts) > 1 else "session.pkl"
            try:
                result = self.session.load_session(filename)
                console.print(f"[green]{result}[/green]")
            except Exception as e:
                console.print(f"[red]Error loading session: {e}[/red]")

        elif cmd == "/exit" or cmd == "/quit" or cmd == "/q":
            console.print("\n[cyan]Exiting session...[/cyan]")
            self.running = False

        else:
            console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            console.print("Type /help to see available commands")


async def run_interactive_session(
    project_root: str,
) -> None:
    """Run an interactive coding session.

    Args:
        project_root: Root directory of the project
    """

    session = InteractiveSession(project_root=project_root)

    # Run interactive CLI
    cli = InteractiveCLI(session)
    await cli.run()
