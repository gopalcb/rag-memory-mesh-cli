import json
import shlex
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from scripts.ensure_ingestion_ui import ensure_ui
from scripts.install_codex_hook import HOOK_LABEL, install_hook


def test_hook_install_preserves_other_hooks_and_is_idempotent(tmp_path):
    hooks = tmp_path / "hooks.json"
    existing = [{"hooks": [{"type": "command", "command": "/path/to/existing"}]}]
    hooks.write_text(json.dumps({"hooks": {"SessionStart": existing}}))
    interpreter = "/path with spaces/.venv/bin/python"

    assert install_hook(hooks, interpreter, 27743)
    assert not install_hook(hooks, interpreter, 27743)
    groups = json.loads(hooks.read_text())["hooks"]["SessionStart"]
    assert groups[0:1] == existing
    handler = groups[1]["hooks"][0]
    assert handler["statusMessage"] == HOOK_LABEL
    assert shlex.split(handler["command"])[0] == interpreter
    assert shlex.split(handler["command"])[-2:] == ["--port", "27743"]


@pytest.mark.parametrize("service", ["rag-memory-ingestion", "another-service"])
def test_ensure_ui_checks_service_identity(tmp_path, service):
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/health"
            data = json.dumps({"service": service, "status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            if service == "rag-memory-ingestion":
                assert ensure_ui(server.server_port, tmp_path)["status"] == "already_running"
            else:
                with pytest.raises(RuntimeError, match="already occupied"):
                    ensure_ui(server.server_port, tmp_path)
        finally:
            server.shutdown()
            thread.join()
