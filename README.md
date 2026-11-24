# RAG Memory Mesh CLI

A local memory service for coding agents. It indexes Python and Markdown so an agent can find an earlier implementation, understand its dependencies, and record new work for the next task. Use the CLI directly or connect the same service over MCP.

## How a task becomes reusable

1. **Decide whether to look up prior work.** The calling agent skips lookup for greetings and self-contained questions. If the user opts out of RAG, search, or embeddings, the agent skips both lookup and recording for that task.
2. **Send a focused query and keywords.** For “make the HTTP client wait longer after a failed attempt,” the agent might send the query “Reuse capped exponential backoff for transient HTTP failures” and the keywords `retry_delay`, `exponential backoff`, and `Retry-After`.
3. **Retrieve evidence.** Chroma finds chunks with similar meaning. SQLite full-text search ranks matches for the supplied terms and phrases. The retriever combines the two rankings and can append a directly related Python helper. Results include excerpts and source locations. A miss returns `no_results` with an empty list.
4. **Implement and verify the task.** The calling agent owns the edits and checks. When memory is allowed, it sends the implementation summary, decisions, verification outcomes, and relevant file changes after completion, including when there was no lookup or no match.
5. **Index the handoff.** The coordinator writes a searchable Markdown summary and indexes any supplied Python and Markdown snapshots. An identical retry reuses the handoff ID and skips sources already indexed. Later tasks can retrieve the summary and the files that explain it.

## Start locally

The pinned dependencies and CI target Python 3.14 on macOS.

```bash
git clone https://github.com/gopalcb/rag-memory-mesh-cli.git
cd rag-memory-mesh-cli
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python rag_cli.py --help
```

The first embedding run downloads the pinned `all-MiniLM-L6-v2` model. After it is cached, set `HF_HUB_OFFLINE=1` for offline embeddings. Ingestion and search work without a chat model. The optional blueprint planner uses Ollama by default.

### Index a project and search it

```bash
.venv/bin/python run_embed.py /absolute/path/to/project
.venv/bin/python rag_cli.py search retry_delay --mode keyword
.venv/bin/python rag_cli.py search "exponential retry delay" --mode hybrid
.venv/bin/python rag_cli.py search "retry policy" --collection research_docs --mode hybrid
```

A folder run recursively indexes supported, nonempty `.py`, `.md`, and `.markdown` files. It skips `__init__.py`, symlinks, and ignored directories such as `.git`, `.venv`, and `node_modules`. The default collections are `reusable_code` and `research_docs`. Use `-k` to set the number of ranked hits.

Python is chunked by symbols and retains AST dependency edges. Markdown is chunked by headings with heading context and source metadata. Embedding vectors live in Chroma. SQLite stores the text index, code relationships, and the manifest that tracks indexed sources.

### Connect an agent over MCP

From the repository root, register the local stdio server with your MCP client. For Codex:

```bash
codex mcp add rag-memory-mesh-cli -- "$PWD/.venv/bin/python" "$PWD/mcp_server.py"
```

To select a custom profile, add `--env RAG_CONFIG_PATH=/absolute/path/profile.yaml` before `--`. Use the same profile for local ingestion and MCP so both see the same index. The agent-facing rules are in [AGENTS.md](AGENTS.md).

| Tool | When to call it |
| --- | --- |
| `search_memory(query, keywords)` | Before a project task when prior work could help. Send a refined query and distinctive keywords or phrases. |
| `record_task_changes(project, task, implementation_summary, verification, files)` | After each completed implementation task where memory is allowed. Supply project-relative paths and what changed. Include final text for relevant Python and Markdown files. Omit content for deleted or unsupported files. |
| `submit_plan`, `run_plan`, `get_plan`, `answer_plan` | To queue and run a source-linked blueprint job, inspect its status, or answer a clarification. |
| `commit_verified_snippet` | To save a small Python snippet with the verification outcome observed by the caller. |

A completion handoff can contain both a source snapshot and path-only entries:

```json
{
  "project": "http-client",
  "task": "Cap retry delays",
  "implementation_summary": "Added exponential backoff capped at eight seconds.",
  "verification": "Passed the retry tests for the first attempt and the cap.",
  "files": [
    {
      "path": "transport/retry.py",
      "change": "Added the delay helper.",
      "content": "def retry_delay(attempt):\n    return min(8, 2 ** attempt)\n"
    },
    {
      "path": "config/legacy-retries.json",
      "change": "Removed the obsolete toggle."
    }
  ]
}
```

### Run a blueprint job from the CLI

With Ollama running and the configured model available, submit a task with enough context to start. Replace `JOB_ID` with the returned ID.

```bash
ollama pull qwen2.5-coder:3b
.venv/bin/python rag_cli.py submit "Plan capped HTTP retries" --context "Python client in transport/retry.py"
.venv/bin/python rag_cli.py run JOB_ID
.venv/bin/python rag_cli.py status JOB_ID
```

If a job returns `needs_context`, continue it with `rag_cli.py answer JOB_ID "your context"`. To research a public documentation URL, pass `--source` during submission. Crawling requires Playwright Chromium and a configured HTTPS host allowlist.

## Profiles and index maintenance

By default the index is under `.rag_memory_data/`. A separate YAML profile selected with `RAG_CONFIG_PATH` can choose storage paths and providers. Relative storage paths resolve from the profile file. Embeddings are local by default, and external model providers require explicit opt-in. A private profile requires user-owned storage with mode `0700`.

| Command | Purpose |
| --- | --- |
| `rag_cli.py doctor` | Check consistency across the vector, lexical, graph, and manifest stores. |
| `rag_cli.py recover` | Replay pending index journal operations. |
| `rag_cli.py ingest PATH --reconcile` | Refresh a folder and remove tracked files that are missing or empty within it. |
| `rag_cli.py backup PATH` / `rag_cli.py restore PATH` | Export or restore a verified snapshot. |

Run these commands with `.venv/bin/python`. Run the test suite with `.venv/bin/python -m pytest -q). For the module layout and process boundaries, see the [architecture](ARCHITECTURE.md) and [code map](code-map.md).
