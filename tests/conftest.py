"""Bounded file-backed child IO for the two real process-lock test suites."""
import os
import subprocess
import time
from contextlib import contextmanager
from itertools import count

import pytest


class BoundedChild:
    def __init__(self, process, stdout_path, stderr_path):
        self.process = process
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    def wait_for_line(self, expected, *, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exited = self.process.poll() is not None
            output = self.stdout_path.read_text(encoding="utf-8")
            if "\n" in output:
                assert output.splitlines()[0] == expected, self.stderr_path.read_text(
                    encoding="utf-8")[-2000:]
                return
            if exited:
                break
            time.sleep(0.01)
        errors = self.stderr_path.read_text(encoding="utf-8")[-2000:]
        raise AssertionError(f"Child startup did not produce {expected!r}: {errors}")

    def communicate(self, *, timeout=15):
        try:
            self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            exc.stderr = self.stderr_path.read_text(encoding="utf-8")[-2000:]
            raise
        return (self.stdout_path.read_text(encoding="utf-8"),
                self.stderr_path.read_text(encoding="utf-8"))


@pytest.fixture
def bounded_child(tmp_path):
    sequence = count()

    @contextmanager
    def start(command, *, env=None):
        number = next(sequence)
        stdout_path = tmp_path / f"child-{number}.stdout"
        stderr_path = tmp_path / f"child-{number}.stderr"
        # Files cannot fill a pipe while the parent waits for a handshake. Read
        # through separate handles so the parent never moves the child's offset.
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8",
        ) as err:
            child_env = {**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"}
            process = subprocess.Popen(command, env=child_env, stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, text=True)
            try:
                yield BoundedChild(process, stdout_path, stderr_path)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
                if process.stdin is not None:
                    process.stdin.close()

    return start
