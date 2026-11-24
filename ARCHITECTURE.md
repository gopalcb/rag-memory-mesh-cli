# RAG Memory Agents architecture

RAG Memory Agents indexes local Python and Markdown, retrieves reusable work for coding agents, and can draft source-linked implementation plans. The calling agent decides when to search, implements the task, and sends completion details when memory is allowed.

The main entry points are [rag_cli.py](rag_cli.py) for local operations, [run_embed.py](run_embed.py) for recursive manual ingestion, and [mcp_server.py](mcp_server.py) for coding-agent tools. Each MCP request starts a bounded CLI worker. The worker selects its storage profile through [config.py](config.py) and takes the index lock before accessing the shared stores.

- **Lookup.** The refined query drives Chroma vector search, while the supplied keyword phrases drive SQLite full-text search. Hybrid ranking combines the two lists and can include direct code dependencies.
- **Completion handoff.** The MCP server validates the request without opening the index. The worker validates the payload, stages the summary and snapshots in the configured Markdown vault, and publishes the directory before invoking incremental ingestion. The stable handoff ID lets a retry reuse the directory and resume indexing pending sources.
- **Planning and research.** The planning coordinator uses the same retriever, stores job progress in SQLite, and can add cached material from the crawler. The configured chat model drafts the blueprint.
- **Storage.** Chroma holds vectors. SQLite holds lexical, graph, manifest, journal, and planning records. The ingestion journal allows recovery when a write is interrupted between stores.

See the [code map](code-map.md) and [end-to-end guide](README.md).
