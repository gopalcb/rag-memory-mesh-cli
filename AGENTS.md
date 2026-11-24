# RAG Memory Agents

Use `rag_cli.py` for local ingestion, retrieval, crawling, and planning jobs. `run_embed.py` also supports embedding-only ingestion. `mcp_server.py` exposes memory lookup, completion handoffs, planning, and verified snippets to coding agents. The optional Flask/Vue console requires a local copy of the ingestion UI package.

## Main-agent memory calls

Skip all memory tools when the user asks to skip search, memory, or embeddings. Skip memory lookup for greetings or self-contained requests. For substantive project work that could benefit from prior work, send a refined query and distinctive keywords to `search_memory`. After every completed implementation task where RAG Memory is allowed, send the implementation summary, decisions, verification and file changes to `record_task_changes` regardless of the lookup outcome. Include relevant final Python and Markdown files and respect any retention restrictions. [Agent memory workflow](README.md#how-a-task-becomes-reusable)

## Workflow

1. **Plan.** Read the requested change and compare it to the live code and existing tests. For cross-component changes, save the phased plan and acceptance checks under `docs/`.
2. **Implement.** Keep existing callers and indexes usable. Keep vector, lexical, graph, and manifest behavior in view when changing ingestion. Planning may write service-owned checkpoints and memory. The calling coding agent owns repository edits and command execution.
3. **Validate.** Run `.venv/bin/python -m pytest -q`. Exercise changed entry points with temporary storage through `RAG_CONFIG_PATH`. For planner changes, check the CLI lifecycle and MCP tool results before relying on either interface. For data changes, exercise the CLI audit and recovery paths as appropriate. [End-to-end guide](README.md)
4. **Document.** Record implementation, changed files, validation evidence, and remaining limits under `docs/`. Keep proposed improvements under `docs/improvements/`.
