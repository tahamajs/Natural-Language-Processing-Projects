"""Interactive CLI mode for multi-turn conversations."""

import asyncio
import os
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.markdown import Markdown
from rich.tree import Tree
from rich.filesize import decimal
from rich.table import Table
from rich import box

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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
        console.print(
            Panel(
                "[bold cyan]Coding Agent[/bold cyan]\n\n"
                f"Project: {self.session.project_root}\n"
                "Type your coding tasks or questions. Use /help for commands.",
                border_style="cyan",
                title="Welcome",
            )
        )

    def _print_help(self) -> None:
        """Print help message."""
        help_text = """[bold]Available Commands:[/bold]

[cyan]/help[/cyan]        - Show this help message
[cyan]/save[/cyan]        - Save current session (Bonus)
[cyan]/load [file][/cyan] - Load session from file (Bonus)
[cyan]/clear[/cyan]       - Clear the terminal screen
[cyan]/history[/cyan]     - Show conversation history
[cyan]/paste[/cyan]       - Enter multi-line paste mode
[cyan]/context[/cyan]     - Show context statistics
[cyan]/mode[/cyan]        - Toggle auto/safe mode
[cyan]/test[/cyan]        - Run test suite
[cyan]/retry[/cyan]       - Rewind last turn
[cyan]/tree[/cyan]        - Show project tree visualization
[cyan]/todos[/cyan]       - Scan for TODO/FIXME comments
[cyan]/complexity [file][/cyan] - Analyze code complexity
[cyan]/doc [file][/cyan]  - Auto-generate documentation
[cyan]/coverage[/cyan]    - Run test coverage analysis
[cyan]/audit[/cyan]       - Security vulnerability scan
[cyan]/persona [mode][/cyan] - Switch agent persona
[cyan]/diagram[/cyan]     - Generate architecture diagram
[cyan]/exit[/cyan]        - Exit the session

[bold]Tips:[/bold]
- The agent remembers your conversation history
- Recently accessed files are tracked automatically
- Use natural language to describe tasks

[bold]Environment variables (HITL):[/bold]
- `CODING_AGENT_AUTO_APPROVE=1` : auto-approve all sensitive tools (use with caution)
- `CODING_AGENT_AUTO_APPROVE_WHITELIST=tool1,tool2` : comma-separated tools to auto-approve
"""
        console.print(Panel(help_text, border_style="blue", title="Help"))

    def _print_goodbye(self) -> None:
        """Print goodbye message."""

        console.print(
            Panel(
                "[bold cyan]Session Ended[/bold cyan]\n\n"
                "Thank you for using Coding Agent!",
                border_style="cyan",
                title="Goodbye",
            )
        )

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
            # Show spinner while agent processes the turn
            with console.status(
                "[bold green]Agent is thinking...[/bold green]", spinner="dots"
            ):
                turn_result = await self.session.process_turn(user_message)

            # Display response with Markdown rendering and syntax highlighting
            console.print(f"\n[bold blue]Agent[/bold blue]:")
            md_response = Markdown(turn_result)
            console.print(
                Panel(
                    md_response,
                    border_style="blue",
                    title="Response",
                )
            )

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

        elif cmd == "/clear":
            # Clear terminal and reprint welcome
            os.system("cls" if os.name == "nt" else "clear")
            self._print_welcome()

        elif cmd == "/history":
            try:
                snapshot = self.session.graph.get_state(self.session.config)
                messages = snapshot.values.get("messages", []) or []
                console.print("\n[bold]Conversation History:[/bold]")
                for msg in messages:
                    mtype = getattr(msg, "type", None)
                    content = getattr(msg, "content", "") or str(msg)
                    if mtype and mtype.lower().startswith("human"):
                        console.print(f"[green]You:[/green] {content}")
                    elif mtype and mtype.lower().startswith("ai"):
                        preview = content[:200] + ("..." if len(content) > 200 else "")
                        console.print(f"[blue]Agent:[/blue] {preview}")
                    else:
                        # ToolMessage or unknown
                        name = getattr(msg, "name", "tool")
                        console.print(f"[dim]Tool ({name}): {len(content)} chars[/dim]")
            except Exception as e:
                console.print(f"[red]Could not retrieve history: {e}[/red]")

        elif cmd == "/paste":
            console.print(
                "[yellow]Entering multi-line mode. Type 'EOF' on a new line to send.[/yellow]"
            )
            lines = []
            while True:
                try:
                    line = await asyncio.to_thread(Prompt.ask, "")
                except EOFError:
                    break
                if line.strip().upper() == "EOF":
                    break
                lines.append(line)

            full_message = "\n".join(lines)
            if full_message.strip():
                await self._process_turn(full_message)
        elif cmd == "/context":
            try:
                state = self.session.graph.get_state(self.session.config)
                msgs = state.values.get("messages", []) or []
                file_content_size = 0
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        content = getattr(m, "content", "")
                        if isinstance(content, str):
                            file_content_size += len(content)

                console.print(
                    Panel(
                        f"Total Messages: [bold]{len(msgs)}[/bold]\n"
                        f"File Content in Memory: [bold]{file_content_size} chars[/bold]\n"
                        f"Thread ID: {self.session.thread_id}",
                        title="Context Inspector",
                        border_style="magenta",
                    )
                )
            except Exception as e:
                console.print(f"[red]Could not inspect context: {e}[/red]")

        elif cmd == "/mode":
            # Toggle runtime auto_mode for safety vs. auto
            try:
                self.session.auto_mode = not getattr(self.session, "auto_mode", False)
                status = (
                    "[bold red]AUTO (Dangerous)[/bold red]"
                    if self.session.auto_mode
                    else "[bold green]SAFE (HITL)[/bold green]"
                )
                console.print(f"Switched to {status} mode.")
            except Exception as e:
                console.print(f"[red]Could not toggle mode: {e}[/red]")

        elif cmd == "/test":
            console.print("[dim]Injecting test command...[/dim]")
            await self._process_turn(
                "Run the full test suite using pytest and report results."
            )

        elif cmd == "/retry":
            try:
                state = self.session.graph.get_state(self.session.config)
                msgs = state.values.get("messages", []) or []
                if len(msgs) > 2:
                    new_msgs = msgs[:-2]
                    try:
                        self.session.graph.update_state(
                            self.session.config, {"messages": new_msgs}
                        )
                        console.print(
                            "[green]Rewound last turn. Try phrasing your request differently.[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Failed to rewind state: {e}[/red]")
                else:
                    console.print("[red]Cannot rewind further.[/red]")
            except Exception as e:
                console.print(f"[red]Retry failed: {e}[/red]")

        elif cmd == "/tree":
            # Create the root of the tree
            tree = Tree(
                f":open_file_folder: [bold]{self.session.project_root.name}[/bold]",
                guide_style="bold bright_blue",
            )

            def build_tree(path, tree_node):
                # Sort: Directories first, then files
                paths = sorted(
                    path.iterdir(), 
                    key=lambda p: (not p.is_dir(), p.name.lower())
                )
                for p in paths:
                    # Skip hidden/git
                    if p.name.startswith(".") or p.name == "__pycache__":
                        continue
                    
                    if p.is_dir():
                        branch = tree_node.add(f":open_file_folder: [bold]{p.name}[/bold]")
                        build_tree(p, branch)
                    else:
                        size = decimal(p.stat().st_size)
                        tree_node.add(f":page_facing_up: {p.name} [dim]({size})[/dim]")

            build_tree(self.session.project_root, tree)
            console.print(tree)

        elif cmd == "/todos":
            console.print("[bold yellow]Scanning for TODOs...[/bold yellow]")
            found_todos = []
            
            # Simple recursive scan
            for path in self.session.project_root.rglob("*.py"):
                if ".git" in path.parts or "__pycache__" in path.parts: continue
                
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                    for i, line in enumerate(lines):
                        if "TODO" in line or "FIXME" in line:
                            clean_line = line.strip()
                            found_todos.append((path.name, i+1, clean_line))
                except Exception:
                    pass

            if found_todos:
                table = Table(title="Tech Debt Detected", box=box.SIMPLE)
                table.add_column("File", style="cyan")
                table.add_column("Line", style="magenta")
                table.add_column("Task", style="yellow")
                
                for fname, line, task in found_todos:
                    table.add_row(fname, str(line), task)
                console.print(table)
                
                # Optional: Prompt to auto-fix?
                console.print("[dim]Tip: Ask agent 'Fix all items in the TODO list'[/dim]")
            else:
                console.print("[green]No TODOs found! Clean code.[/green]")

        elif cmd.startswith("/complexity"):
            # Usage: /complexity src/analyzer.py
            parts = command.split()
            target = parts[1] if len(parts) > 1 else "."
            
            try:
                # Run radon cc (Cyclomatic Complexity)
                cmd = ["radon", "cc", target, "-a", "-s"]
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.session.project_root)
                console.print(Panel(res.stdout, title="Cyclomatic Complexity Report", border_style="magenta"))
            except FileNotFoundError:
                console.print("[red]Radon not installed. Run `pip install radon`[/red]")
            except Exception as e:
                console.print(f"[red]Error running complexity analysis: {e}[/red]")

        elif cmd.startswith("/doc"):
            # Usage: /doc model/bigram.py
            parts = command.split()
            if len(parts) < 2:
                console.print("[red]Usage: /doc <filename>[/red]")
            else:
                filename = parts[1]
                console.print(f"[cyan]Generating documentation for {filename}...[/cyan]")
                prompt = (
                    f"Please read '{filename}' and add high-quality Python docstrings "
                    "to all classes and methods that are missing them. "
                    "Do not change any logic, just add comments."
                )
                await self._process_turn(prompt)

        elif cmd == "/coverage":
            console.print("[cyan]Running test coverage analysis...[/cyan]")
            try:
                # Run pytest with coverage
                cmd = ["pytest", "--cov=.", "--cov-report=term-missing"]
                res = subprocess.run(
                    cmd, 
                    cwd=self.session.project_root, 
                    capture_output=True, 
                    text=True
                )
                
                # Display output in a panel
                if res.returncode == 0:
                    style = "green"
                    title = "Coverage: PASS"
                else:
                    style = "yellow" 
                    title = "Coverage: FAIL/INCOMPLETE"
                    
                console.print(Panel(
                    res.stdout,
                    title=title,
                    border_style=style,
                    subtitle="Missing lines shown in right column"
                ))
            except FileNotFoundError:
                console.print("[red]pytest-cov not installed. Run `pip install pytest-cov`[/red]")

        elif cmd == "/audit":
            console.print("[bold red]Scanning for security vulnerabilities...[/bold red]")
            try:
                # Run bandit recursively
                cmd = ["bandit", "-r", ".", "-f", "screen"]
                res = subprocess.run(
                    cmd, 
                    cwd=self.session.project_root, 
                    capture_output=True, 
                    text=True
                )
                
                if "No issues identified" in res.stdout:
                    console.print("[bold green]✓ Project is Secure (No common vulnerabilities found)[/bold green]")
                else:
                    console.print(Panel(res.stdout or res.stderr, title="Security Alert", border_style="red"))
            except FileNotFoundError:
                console.print("[red]Bandit not installed. Run `pip install bandit`[/red]")

        elif cmd.startswith("/persona"):
            # Usage: /persona teacher
            try:
                mode = command.split()[1]
                msg = self.session.set_system_prompt(mode)
                console.print(f"[green]{msg}[/green]")
            except IndexError:
                console.print("[yellow]Usage: /persona [default|teacher|architect|junior][/yellow]")

        elif cmd == "/diagram":
            console.print("[cyan]Generating Class Diagram...[/cyan]")
            prompt = (
                "Analyze all Python files in the 'src' and 'model' directories. "
                "Generate a Mermaid.js class diagram syntax (graph TD or classDiagram) "
                "showing the relationships between classes. Output ONLY the mermaid code block."
            )
            await self._process_turn(prompt)

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
