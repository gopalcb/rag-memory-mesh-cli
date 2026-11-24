# Ingestion console behavior

The console manages local embedding runs while leaving planning and retrieval in the CLI. A user enters a local file or folder for a one-time run, or enters a folder for ongoing indexing. The Flask service accepts local requests and calls the existing embedding pipeline. [CLI documentation](README.md#run-a-blueprint-job-from-the-cli)

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Form | Native form and shared Vue source form | `ingestion_ui/templates/index.html`, `ingestion_ui/static/ingestion.js` | Index once or Watch folder | Keyboard and browser flow |
| Scrollbar | Global stylesheet | `ingestion_ui/static/ingestion.css` | Watch list and activity table overflow | Computed style and rendered viewport |
| Feedback | Inline field errors, job rows and status regions | `ingestion_ui/templates/index.html`, `ingestion_ui/static/ingestion.js` | Pending, success, failure, disconnected | Browser failure and recovery |
| Job and watch updates | Flask routes, durable queue and worker | `ingestion_ui/__init__.py`, `ingestion_ui/queue.py`, `ingestion_ui/worker.py` | One-time job, automatic watch job, manual retry, pause and resume | API and real browser flows |

## State behavior

- Confirm jobs and watch changes after the server accepts them. Keep a path available for correction when entry validation fails.
- Record each ingestion attempt in recent activity. Show a reason on failed runs and offer retry as a new attempt.
- A watch can be active, unavailable or paused. Pausing stops future automatic scheduling. Active watches use a persisted folder snapshot after restart.
- Keep the last good dashboard visible when a refresh fails. Expose reconnect feedback and a retry button.
