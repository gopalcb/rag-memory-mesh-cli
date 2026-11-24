import fcntl
import hashlib
import os
import stat
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from config import settings


@contextmanager
def index_lock():
    """Serialize operations sharing either configured persistent database."""
    directory = Path("/tmp").resolve() / f"rag-memory-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("Unsafe index lock directory")
    paths = (settings.storage.chroma_dir.resolve(), settings.storage.sqlite_db_path.resolve())
    keys = sorted({hashlib.sha256(os.fsencode(path)).hexdigest() for path in paths})
    deadline = time.monotonic() + settings.runtime.lock_timeout_seconds
    # Consistent ordering also prevents lock inversion across misconfigured profiles.
    with ExitStack() as stack:
        for key in keys:
            fd = os.open(directory / key, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, "a"))
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Index is busy. Retry when the current operation completes.") from None
                    time.sleep(0.05)
        yield
