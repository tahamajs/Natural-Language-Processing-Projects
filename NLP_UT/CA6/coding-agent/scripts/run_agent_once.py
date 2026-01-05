"""Run the InteractiveSession for a single turn and print the response.

This script is intended for smoke-testing the agent non-interactively.
It performs one `process_turn` call and exits.
"""
import asyncio
import sys
from pathlib import Path

# Ensure package import works when running from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coding_agent.agent.session import InteractiveSession


async def main(project_root: str, user_message: str):
    session = InteractiveSession(project_root=project_root)
    resp = await session.process_turn(user_message)
    print("AGENT RESPONSE:")
    print(resp)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_agent_once.py <project_root> <user_message>")
        sys.exit(1)
    proj = sys.argv[1]
    msg = " ".join(sys.argv[2:])
    asyncio.run(main(proj, msg))

