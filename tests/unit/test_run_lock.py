from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def lock_type():
    name = "shu_insight.services.run_lock"
    assert importlib.util.find_spec(name) is not None, "OS run lock is not implemented"
    return importlib.import_module(name).RunLock


def child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def child_lock(bounded_child, path: Path, *, hold: bool = False):
    code = """
import sys
from pathlib import Path
import shu_insight.services.run_lock as module
assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
lock = module.RunLock(Path(sys.argv[1]))
acquired = lock.acquire()
print(acquired, flush=True)
if acquired and sys.argv[3] == 'hold':
    sys.stdin.readline()
lock.release()
"""
    return bounded_child(
        [sys.executable, "-u", "-c", code, str(path), str(ROOT / "src"),
         "hold" if hold else "once"],
        env=child_env(),
    )


def test_nonblocking_lock_preserves_first_byte_and_releases_on_exception(tmp_path):
    run_lock = lock_type()
    path = tmp_path / "run.lock"
    first = run_lock(path)
    second = run_lock(path)
    assert first.acquire()
    try:
        assert not second.acquire()
    finally:
        first.release()
    # Windows byte locks prohibit a separate handle reading that byte while held.
    assert path.read_bytes() == b"\0"
    with pytest.raises(RuntimeError):
        with second:
            assert second.acquired
            raise RuntimeError("local failure")
    assert first.acquire()
    first.release()
    first.release()
    assert path.read_bytes() == b"\0"


@pytest.mark.parametrize("terminate", [False, True])
def test_independent_process_contention_and_os_exit_release(tmp_path, terminate, bounded_child):
    lock_type()
    path = tmp_path / "run.lock"
    with child_lock(bounded_child, path, hold=True) as first:
        first.wait_for_line("True")
        with child_lock(bounded_child, path) as second:
            output, errors = second.communicate()
            assert second.process.returncode == 0, errors
            assert output.strip() == "False"
        if terminate:
            first.process.kill()
        else:
            first.process.stdin.write("release\n")
            first.process.stdin.flush()
        _, errors = first.communicate()
        if not terminate:
            assert first.process.returncode == 0, errors
        with child_lock(bounded_child, path) as third:
            output, errors = third.communicate()
            assert third.process.returncode == 0, errors
            assert output.strip() == "True"


@pytest.mark.parametrize("failure", ["startup_timeout", "startup_exit", "communication_timeout"])
def test_child_failure_is_bounded_reports_stderr_and_reaps_process(bounded_child, failure):
    code = "import sys, time; print('diagnostic', file=sys.stderr, flush=True); "
    if failure == "startup_exit":
        code += "sys.exit(7)"
    elif failure == "communication_timeout":
        code += "print('entered', flush=True); time.sleep(60)"
    else:
        code += "time.sleep(60)"
    expected = subprocess.TimeoutExpired if failure == "communication_timeout" else AssertionError
    with (
        pytest.raises(expected) as error,
        bounded_child([sys.executable, "-u", "-c", code]) as child,
    ):
        if failure == "communication_timeout":
            child.wait_for_line("entered")
            child.communicate(timeout=0.2)
        else:
            child.wait_for_line("entered", timeout=0.5)
    assert child.process.poll() is not None
    if failure != "communication_timeout":
        assert "diagnostic" in str(error.value)
    else:
        assert "diagnostic" in error.value.stderr


def test_child_large_stderr_does_not_block_startup_handshake(bounded_child):
    code = ("import sys; sys.stderr.write('x' * 200000); sys.stderr.flush(); "
            "print('entered', flush=True)")
    with bounded_child([sys.executable, "-u", "-c", code]) as child:
        child.wait_for_line("entered")
        output, errors = child.communicate()
        assert output.strip() == "entered"
        assert len(errors) == 200000
    assert child.process.returncode == 0
