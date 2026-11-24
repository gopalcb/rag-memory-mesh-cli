import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import fcntl
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config import settings
from crawler.fleet import CrawlFleet
from planning.blueprint import BlueprintPlanner, Evidence
from store.hybrid_store import HybridRetriever
from store.job_store import JobStore


CONTEXT_QUESTION = (
    "What repository or stack, target files, and constraints should this plan account for? "
    "If the task is standalone, say so and provide any runtime or dependency requirements."
)


class PlanState(TypedDict, total=False):
    task: str
    context: str
    urls: list[str]
    evidence: list[dict]
    crawl_outcomes: list[dict]
    artifact: dict


class JobRunningError(RuntimeError):
    pass


class PlanningCoordinator:
    def __init__(
        self, jobs: JobStore | None = None, retriever: HybridRetriever | None = None,
        crawler: CrawlFleet | None = None, planner: BlueprintPlanner | None = None,
    ):
        self.jobs = jobs or JobStore()
        self.retriever = retriever
        self.crawler = crawler
        self.planner = planner or BlueprintPlanner()
        db = Path(self.jobs.db_path or settings.storage.sqlite_db_path).resolve()
        self.checkpoint_path = db.with_suffix(".checkpoints.sqlite3")
        self.lock_dir = db.parent / "planning-locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock_job(self, job_id: str):
        with (self.lock_dir / f"{UUID(job_id).hex}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JobRunningError(f"Job {job_id} is already being processed") from exc
            yield

    def submit(
        self, task: str, context: str = "", sources: list[str] | None = None,
        skills_needed: list[str] | None = None,
    ) -> str:
        if not task.strip() or len(task) > 12000:
            raise ValueError("Task must contain 1 to 12000 characters")
        if len(context) > 40000:
            raise ValueError("Context must contain at most 40000 characters")
        urls = [target.url_str for target in CrawlFleet.normalize_targets(sources or [])]
        return self.jobs.create_job(task.strip(), context.strip(), skills_needed, research_urls=urls)

    def snapshot(self, job_id: str) -> dict:
        item = self.jobs.get_job_snapshot(job_id)
        if item is None:
            raise ValueError(f"Unknown planning job {job_id}")
        if item["final_plan"]:
            try:
                item["blueprint"] = json.loads(item["final_plan"])
            except json.JSONDecodeError:
                item["blueprint"] = None
        return item

    def clarify(self, state: PlanState) -> dict:
        if state["context"]:
            return {}
        return {"context": interrupt(CONTEXT_QUESTION)}

    def collect_evidence(self, task: str, source_urls: list[str] | None = None) -> list[Evidence]:
        self.retriever = self.retriever or HybridRetriever()
        items = []
        for collection_kind, collection in (
            ("code", settings.collections.code), ("docs", settings.collections.docs)
        ):
            hits = []
            if collection_kind == "docs" and source_urls:
                for url in source_urls:
                    hits.extend(self.retriever.search(
                        task, options={"collection_name": collection, "k": 1,
                                       "where": {"source_url": url}, "mode": "vector", "max_distance": 2},
                    ))
            hits.extend(self.retriever.search(task, options={"collection_name": collection, "k": 3}))
            seen = set()
            for hit in hits:
                meta = hit.metadata
                if meta["chunk_id"] in seen:
                    continue
                seen.add(meta["chunk_id"])
                items.append(Evidence(
                    id=f"E{len(items) + 1}", collection=collection,
                    source=meta.get("source", ""),
                    location=meta.get("source_url") or meta.get("file_path", ""),
                    content=hit.document.page_content[:2400],
                ))
                if len(seen) == 6:
                    break
        return items

    def research(self, state: PlanState) -> dict:
        outcomes = []
        sources = state["urls"]
        if sources:
            self.crawler = self.crawler or CrawlFleet(jobs=self.jobs)
            report = asyncio.run(self.crawler.crawl_urls(sources))
            outcomes = [outcome.model_dump() for outcome in report.outcomes]
        stale = {item["url"] for item in outcomes if item["status"] == "failed"}
        evidence = [
            item for item in self.collect_evidence(state["task"], [url for url in sources if url not in stale])
            if item.location not in stale
        ]
        return {"evidence": [item.model_dump() for item in evidence], "crawl_outcomes": outcomes}

    def draft(self, state: PlanState) -> dict:
        evidence = [Evidence.model_validate(item) for item in state["evidence"]]
        blueprint = self.planner.draft(state["task"], state["context"], evidence)
        return {"artifact": {
            **blueprint.model_dump(),
            "grounding": "cited_sources" if any(step.evidence_ids for step in blueprint.steps) else "proposal",
            "sources": [item.model_dump() | {"excerpt_sha256": item.excerpt_sha256} for item in evidence],
            "crawl_outcomes": state["crawl_outcomes"],
        }}

    def graph(self, checkpointer: SqliteSaver):
        graph = StateGraph(PlanState)
        graph.add_node("clarify", self.clarify)
        graph.add_node("research", self.research)
        graph.add_node("draft", self.draft)
        graph.add_edge(START, "clarify")
        graph.add_edge("clarify", "research")
        graph.add_edge("research", "draft")
        graph.add_edge("draft", END)
        return graph.compile(checkpointer=checkpointer)

    def advance(self, job_id: str, answer: str | None = None) -> dict:
        with self.lock_job(job_id), SqliteSaver.from_conn_string(str(self.checkpoint_path)) as checkpointer:
            job = self.snapshot(job_id)
            if job["status"] == "completed":
                return job
            graph = self.graph(checkpointer)
            config = {"configurable": {"thread_id": job_id}}
            state = graph.get_state(config)
            if answer is not None and (not answer.strip() or len(answer) > 40000 or not state.interrupts):
                raise ValueError("A reply requires a nonempty answer and a paused job")
            saved_answer = False
            if state.interrupts and answer is None:
                last_message = job["messages"][-1] if job["messages"] else None
                if last_message and last_message["message_type"] == "answer":
                    answer = last_message["content"]
                    saved_answer = True
                else:
                    if job["status"] != "needs_context":
                        self.record_question(job_id, state.interrupts[0].value)
                    return self.snapshot(job_id)

            if answer is not None:
                if not saved_answer:
                    self.jobs.post_message(job_id, "main_agent", "answer", answer.strip(), "running")
                request = Command(resume=answer.strip())
            elif state.values:
                request = None
                failed_nodes = {"clarify": START, "research": "clarify", "draft": "research"}
                for pending in state.tasks:
                    if pending.error and pending.name in failed_nodes:
                        graph.update_state(config, {}, as_node=failed_nodes[pending.name])
                        break
            else:
                request = {
                    "task": job["task"], "context": job["codebase_context"],
                    "urls": job["research_urls"],
                }

            self.jobs.update_job_status(job_id, "running", "planning", error_message="")
            try:
                result = graph.invoke(request, config) if request is not None or state.next else state.values
                if result.get("__interrupt__"):
                    self.record_question(job_id, result["__interrupt__"][0].value)
                else:
                    self.jobs.update_job_status(
                        job_id, "completed", "done", final_plan=json.dumps(result["artifact"]),
                    )
            except Exception as exc:
                self.jobs.update_job_status(job_id, "failed", "planning", error_message=str(exc))
            return self.snapshot(job_id)

    def record_question(self, job_id: str, question: str) -> None:
        self.jobs.post_message(job_id, "coordinator", "question", question, "needs_context")
        self.jobs.update_job_status(job_id, "needs_context", "clarify")
