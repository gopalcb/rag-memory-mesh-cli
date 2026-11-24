import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


SCRIPT = Path(__file__).resolve().with_name("ensure_ingestion_ui.py")
HOOK_LABEL = "Check RAG Memory ingestion UI"


def install_hook(hook_file: Path, python: str, port: int, config: Path | None = None) -> bool:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    args = [python, str(SCRIPT), "--hook", "--port", str(port)]
    if config is not None:
        args.extend(["--config", str(config)])
    handler = {
        "type": "command",
        "command": shlex.join(args),
        "statusMessage": HOOK_LABEL,
        "timeout": 12,
    }
    contents = json.loads(hook_file.read_text(encoding="utf-8")) if hook_file.exists() else {}
    groups = contents.setdefault("hooks", {}).setdefault("SessionStart", [])
    for group in groups:
        handlers = group["hooks"]
        for index, existing in enumerate(handlers):
            if existing.get("statusMessage") != HOOK_LABEL:
                continue
            if existing == handler:
                return False
            handlers[index] = handler
            break
        else:
            continue
        break
    else:
        groups.append({"matcher": "startup|resume", "hooks": [handler]})

    hook_file.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=hook_file.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            json.dump(contents, output, indent=2)
            output.write("\n")
            output.close()
            if hook_file.exists():
                temp.chmod(hook_file.stat().st_mode & 0o777)
            temp.replace(hook_file)
        finally:
            temp.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the user-level Codex ingestion startup hook")
    parser.add_argument("--port", type=int, default=27743)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--codex-home", type=Path, default=Path(os.getenv("CODEX_HOME", Path.home() / ".codex")))
    args = parser.parse_args()
    config = args.config.expanduser().resolve() if args.config else None
    if config is not None and not config.is_file():
        parser.error(f"Storage configuration is not a file: {config}")
    hook_file = args.codex_home.expanduser() / "hooks.json"
    installed = install_hook(hook_file, sys.executable, args.port, config)
    print(f"{'Installed' if installed else 'Already configured'} {hook_file}")
    print("Review and trust the hook in Codex with /hooks.")


if __name__ == "__main__":
    main()
