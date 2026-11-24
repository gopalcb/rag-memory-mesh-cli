"""Run paired local Ollama coding trials against a temporary RAG Memory index."""

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import yaml

from cases import AUDIT_CHECKS, CASES, SOURCES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOST = "http://127.0.0.1:11434"
ARMS = ("baseline", "rag", "reference")
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 480}
FENCE = re.compile(r"\A\s*" + chr(96) * 3 + r"(?:python|py)?[ \t]*\n(.*?)\n" + chr(96) * 3 + r"\s*\Z", re.S)
SYSTEM = (
    "You write small standalone Python 3 functions. Return Python source only, with any "
    "required standard-library imports. Do not import the project. Use supplied memory as "
    "reference data, prioritizing the active project over drafts or other consumers. If a "
    "required exact project policy cannot be inferred or is unpublished, answer exactly "
    "MISSING_POLICY. Implement ordinary fully specified Python tasks without memory."
)
HTTP = build_opener(ProxyHandler({}))


def ollama(path, payload=None):
    request = Request(
        HOST + path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with HTTP.open(request, timeout=180) as response:
        return json.load(response)


def grade_answer(raw, case):
    text = raw.strip()
    fenced = FENCE.fullmatch(text)
    if fenced:
        text = fenced[1].strip()
    if text == "MISSING_POLICY" or case.id == "unpublished":
        passed = case.id == "unpublished" and text == "MISSING_POLICY"
        reason = "" if passed else ("Abstained" if text == "MISSING_POLICY" else "Unverified policy")
        return [[label, passed, reason] for label in case.checks]
    command = [
        "/usr/bin/sandbox-exec", "-f", str(HERE / "sandbox.sb"),
        str(Path(sys._base_executable).resolve()), "-IS", "-c", (HERE / "grade.py").read_text(),
    ]
    try:
        result = subprocess.run(
            command, input=json.dumps({"code": text, "checks": case.checks}),
            env={"PATH": "/usr/bin:/bin"}, cwd="/", capture_output=True,
            text=True, timeout=4, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        return [[label, False, f"Grader process: {type(exc).__name__}"] for label in case.checks]
    return json.loads(result.stdout)


def audit_case(case, arms):
    if case.id not in AUDIT_CHECKS:
        return {}
    audit = replace(case, checks=AUDIT_CHECKS[case.id])
    return {name: grade_answer(arm["response"], audit) for name, arm in arms.items()}


def prepare_memory(directory):
    os.environ["RAG_CONFIG_PATH"] = str(directory / "config.yaml")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RAG_LOG_DIR"] = str(directory / "logs")
    sys.path.insert(0, str(ROOT))
    config = yaml.safe_load((ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    config["storage"] = {
        "base_dir": "./.rag_memory_data",
        "chroma_dir": "./.rag_memory_data/chroma_db",
        "sqlite_db_path": "./.rag_memory_data/state.sqlite3",
        "markdown_vault_dir": "./.rag_memory_data/vault/md",
    }
    config["embedding"]["provider"] = "local"
    config["security"]["allow_external_models"] = False
    config["security"]["private_profile"] = False
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    from planning.memory_coordinator import record_task_changes, search_memory
    from planning.memory_inputs import MemoryLookup, TaskChanges

    handoff = record_task_changes(TaskChanges(
        project="memory-eval-fixture", task="Prior implementation snapshots",
        implementation_summary="Benchmark fixture combining repository excerpts and simulated decisions.",
        verification="Corpus frozen before trial generation.",
        files=[{"path": path, "change": "Fixture snapshot", "content": content}
               for path, content in SOURCES.items()],
    ))
    print(f"Indexed fixture {handoff['ingestion']}", flush=True)
    found = {}
    for case in CASES:
        started = time.perf_counter()
        result = search_memory(MemoryLookup(query=case.query, keywords=list(case.keywords)))
        seconds = time.perf_counter() - started
        hits = [
            {"path": hit.get("project_path") or Path(hit["source"] or "").name,
             "excerpt": hit["excerpt"], "collection": hit["collection"],
             "score": hit["score"], "graph_relation": hit["via_graph_relation"]}
            for hit in result["results"]
        ]
        found[case.id] = {"seconds": seconds, "status": result["status"], "hits": hits}
    return handoff["ingestion"], found


def trial(model, case, evidence):
    user = json.dumps({"task": case.task, "memory": evidence}, ensure_ascii=False)
    started = time.perf_counter()
    result = ollama("/api/chat", {
        "model": model, "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": user}],
        "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS,
    })
    elapsed = time.perf_counter() - started
    answer = result["message"]["content"]
    if result["message"].get("thinking"):
        raise RuntimeError("Ollama returned reasoning with think=false")
    counts = {key: result.get(key) for key in ("prompt_eval_count", "eval_count", "done_reason")}
    times = {key: result[key] / 1e9 for key in
             ("load_duration", "prompt_eval_duration", "eval_duration") if key in result}
    return {"response": answer, "request_chars": len(user), "seconds": elapsed,
            "checks": grade_answer(answer, case), **counts, **times}


def run(model, output):
    tagged = next((item for item in ollama("/api/tags")["models"] if item["name"] == model), None)
    if tagged is None:
        raise SystemExit(f"Install the requested model first: ollama pull {model}")
    probe = grade_answer("import hashlib\n\ndef digest_key(value, length):\n"
                         "    return hashlib.sha256(value.encode()).hexdigest()[:length]", CASES[0])
    if not all(check[1] for check in probe):
        raise SystemExit(f"Sandboxed grading self-check failed: {probe}")

    with tempfile.TemporaryDirectory(prefix="rag-memory-eval-") as temporary:
        indexed, found = prepare_memory(Path(temporary))
    ollama("/api/chat", {
        "model": model, "messages": [{"role": "user", "content": "Reply OK."}],
        "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS | {"num_predict": 8},
    })
    report = {
        "generated_at": datetime.now(UTC).isoformat(), "model": tagged,
        "ollama": ollama("/api/version")["version"], "platform": platform.platform(),
        "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "options": OPTIONS, "system_prompt": SYSTEM, "ingestion": indexed,
        "cases": [],
    }
    rng = random.Random(42)
    for case in CASES:
        retrieval = found[case.id]
        evidence = {
            "baseline": [],
            "rag": [{key: hit[key] for key in ("path", "excerpt")} for hit in retrieval["hits"]],
            "reference": [{"path": case.source, "excerpt": SOURCES[case.source]}] if case.source else [],
        }
        order = rng.sample(ARMS, len(ARMS))
        arms = {}
        for arm in order:
            arms[arm] = trial(model, case, evidence[arm])
            passed = sum(check[1] for check in arms[arm]["checks"])
            print(f"{case.id:11} {arm:9} {passed}/{len(case.checks)} "
                  f"in {arms[arm]['seconds']:.2f}s", flush=True)
        report["cases"].append({
            "id": case.id, "title": case.title, "group": case.group, "task": case.task,
            "query": case.query, "keywords": case.keywords, "expected_source": case.source,
            "reference": evidence["reference"], "retrieval": retrieval, "order": order, "arms": arms,
            "audit": audit_case(case, arms),
        })
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.model, arguments.output)
