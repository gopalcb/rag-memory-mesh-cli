"""Expose RAG memory and planning tools over local MCP stdio."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from planning.memory_inputs import (
    ChangedFiles,
    Detail,
    Keywords,
    MemoryLookup,
    ShortText,
    TaskChanges,
)


server = FastMCP("RAG Memory Agents")
CLI = Path(__file__).resolve().with_name("rag_cli.py")
WORKER_TIMEOUT_SECONDS = 300


async def invoke_cli(command: str, **arguments: Any) -> dict[str, Any]:
    """Run a bounded worker using the active storage profile."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(CLI), "_mcp", stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps({"command": command, **arguments}).encode("utf-8")),
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncio.CancelledError) as error:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.communicate()
        if isinstance(error, TimeoutError):
            raise RuntimeError("Planning worker exceeded its 300-second budget. Inspect the job and retry.") from None
        raise
    if stdout:
        return json.loads(stdout)
    raise RuntimeError(stderr.decode("utf-8", errors="replace").strip()[-1000:] or "CLI worker failed")


@server.tool(structured_output=True, annotations=ToolAnnotations(
    readOnlyHint=True, idempotentHint=True, openWorldHint=False,
))
async def search_memory(query: Detail, keywords: Keywords) -> dict[str, Any]:
    """Look up prior work when it could help with a substantive project task.

    Skip RAG Memory tools when the user asks to skip search, memory, or
    embeddings. Skip lookup for greetings or self-contained requests. Send a
    refined standalone query and distinctive keywords or identifiers.
    An empty lookup returns no_results. Use record_task_changes after completing
    an implementation task whenever RAG Memory is allowed.
    """
    lookup = MemoryLookup(query=query, keywords=keywords)
    return await invoke_cli("search-memory", **lookup.model_dump())


@server.tool(structured_output=True, annotations=ToolAnnotations(
    destructiveHint=False, idempotentHint=True, openWorldHint=False,
))
async def record_task_changes(
    project: ShortText,
    task: ShortText,
    implementation_summary: Detail,
    verification: Detail,
    files: ChangedFiles | None = None,
) -> dict[str, Any]:
    """Record details after every completed implementation task where RAG Memory is allowed.

    Send the handoff after a hit, a miss, or a task that required no lookup.
    Include decisions, limitations, observed verification, and file paths with
    what changed. Attach final text for relevant Python and Markdown files,
    including new docs. Use path-only entries for deletions and unsupported
    files. Respect any user restrictions on retention.
    """
    changes = TaskChanges(
        project=project, task=task, implementation_summary=implementation_summary,
        verification=verification, files=files or [],
    )
    return await invoke_cli("record-task-changes", **changes.model_dump())


@server.tool(structured_output=True)
async def submit_plan(task: str, context: str = "", sources: list[str] | None = None) -> dict[str, Any]:
    """Queue a blueprint job when the user wants a plan and allows retrieval.

    Use run_plan to start it. Use search_memory for a routine prior-work lookup.
    """
    return await invoke_cli("submit", task=task, context=context, source=sources or [])


@server.tool(structured_output=True)
async def run_plan(job_id: str) -> dict[str, Any]:
    """Advance a planning job to a clarification interrupt or a finished blueprint."""
    return await invoke_cli("run", job_id=job_id)


@server.tool(structured_output=True, annotations=ToolAnnotations(
    readOnlyHint=True, idempotentHint=True, openWorldHint=False,
))
async def get_plan(job_id: str) -> dict[str, Any]:
    """Read a job's status, mailbox, and completed blueprint."""
    return await invoke_cli("status", job_id=job_id)


@server.tool(structured_output=True)
async def answer_plan(job_id: str, answer: str) -> dict[str, Any]:
    """Respond to a clarification and resume the same planning job."""
    return await invoke_cli("answer", job_id=job_id, answer=answer)


@server.tool(structured_output=True)
async def commit_verified_snippet(name: str, code: str, verification: str) -> dict[str, Any]:
    """Save a Python snippet with the verification result observed by the caller."""
    return await invoke_cli("commit-verified-snippet", name=name, code=code, verification=verification)


if __name__ == "__main__":
    server.run(transport="stdio")
