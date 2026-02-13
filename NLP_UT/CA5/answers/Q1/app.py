from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Q1_SRC = PROJECT_ROOT / "answers" / "Q1" / "src"
if str(Q1_SRC) not in sys.path:
    sys.path.append(str(Q1_SRC))

from q1_pipeline import DEFAULT_INDEX_PATH, run_pipeline


@cl.on_chat_start
async def on_chat_start() -> None:
    message = (
        "Legal Retrieval Assistant is ready. "
        "Ask your legal question and I will answer using indexed evidence."
    )
    await cl.Message(content=message).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    response = await cl.make_async(run_pipeline)(
        message.content,
        k=10,
        top_n=3,
        index_path=DEFAULT_INDEX_PATH,
    )

    answer = response.get("answer", "No answer was generated.")
    timings = response.get("timings", {})
    intent = response.get("intent", "unknown")
    contexts = response.get("contexts", [])

    details = [
        f"Intent: {intent}",
        f"Total latency: {timings.get('total', 0.0):.3f}s",
        f"Retrieved contexts: {len(contexts)}",
    ]
    final_text = "\n".join(details) + "\n\n" + str(answer)

    elements = []
    for idx, context in enumerate(contexts, start=1):
        source_name = str(context.get("metadata", {}).get("title", f"Source {idx}"))
        source_text = str(context.get("text", ""))
        elements.append(cl.Text(name=f"Source {idx}: {source_name}", content=source_text, display="inline"))

    outgoing = cl.Message(content=final_text, elements=elements)
    await outgoing.send()
