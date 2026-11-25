from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO


class RunLock:
    """Nonblocking OS lock; the stable file is never unlinked or PID-inspected."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        return self._file is not None

    def acquire(self) -> bool:
        if self.acquired:
            return True
        # Append mode preserves an existing lock file. Initialize byte zero
        # before locking so Windows always locks a real byte, even on first use.
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        except BaseException:
            handle.close()
            raise
        self._file = handle
        return True

    def release(self) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
