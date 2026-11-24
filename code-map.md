# Code map

| Path | Ownership |
| --- | --- |
| [mcp_server.py](mcp_server.py) | Declares the stdio tool inputs and sends requests to a bounded CLI worker through `invoke_cli`. |
| [rag_cli.py](rag_cli.py) and [run_embed.py](run_embed.py) | Own local commands and acquire the shared index lock. The CLI worker dispatches memory and planning requests. |
| [planning/memory_inputs.py](planning/memory_inputs.py) | Defines `MemoryLookup`, `ChangedFile`, and `TaskChanges`. Validates query terms, relative paths, and snapshot collisions without starting the index. |
| [planning/memory_coordinator.py](planning/memory_coordinator.py) | `search_memory` returns ranked excerpts and snapshot locations. `record_task_changes` validates source limits, stages the handoff, and invokes ingestion. |
| [embedding/loaders.py](embedding/loaders.py) and [embedding/pipeline.py](embedding/pipeline.py) | Share file eligibility between handoffs and directory scans, then chunk, embed, and journal incremental updates. |
| [store/hybrid_store.py](store/hybrid_store.py) and [store/runtime.py](store/runtime.py) | Combine vector and lexical ranks, attach direct dependencies, and serialize access to the configured stores. |
| [planning/coordinator.py](planning/coordinator.py), [crawler/fleet.py](crawler/fleet.py), and [config.py](config.py) | Own planning jobs, research fetches, and profile configuration. |
| [tests/test_embedding.py](tests/test_embedding.py) and [tests/test_production.py](tests/test_production.py) | Cover snapshot validation, interrupted ingestion, lookup, retries, and stdio MCP dispatch. |

See [architecture](ARCHITECTURE.md) for the request and storage flow.
