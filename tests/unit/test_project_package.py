"""The teammate ZIP contains source and reviewable fixtures, never local state."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "package_project.py"


def package_api():
    assert SCRIPT.is_file(), "the bounded project packager has not been implemented"
    spec = importlib.util.spec_from_file_location("project_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    files = {
        "app.py": "print('app')\n",
        "pyproject.toml": "[project]\nname='demo'\n",
        "README.md": "readme",
        ".gitignore": ".env\n",
        ".env.example": "EXAMPLE=\n",
        "run_local.ps1": "python app.py",
        "src/shu_insight/__init__.py": "",
        "src/shu_insight/ui/home.py": "# source\n",
        "config/sources.toml": "# sources\n",
        "config/taxonomy.toml": "# taxonomy\n",
        "config/ranking.toml": "# ranking\n",
        "scripts/import_official_preload.py": "# offline import\n",
        "scripts/package_project.py": "# packager\n",
        "docs/组员协作与运行.md": "组员运行说明",
        "data/official/preload-2026-10-25.json": '{"entries": []}',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_package_includes_sources_and_test_fixtures_but_excludes_local_state(project, tmp_path):
    allowed = {
        "scripts/run_evaluation.py", "tests/conftest.py", "tests/unit/test_something.py",
        "tests/fixtures/sites/list.html", "tests/fixtures/db/old.sql",
        "tests/fixtures/annotations/mini.jsonl", "data/demo/notices.json",
        "data/official/snapshot-2026-09-24.json",
    }
    excluded = {
        ".env", ".git/config", ".venv/private.py", ".codex/session.json",
        ".agents/config.toml", ".streamlit/secrets.toml", "data/shu_insight.sqlite3",
        "data/backups/old.sqlite3", "data/local/note.jsonl", "artifacts/source.py",
        "src/shu_insight/__pycache__/cached.py", "src/shu_insight/.env",
        "src/shu_insight/secret.sqlite3", "config/private.json", "scripts/private.py",
        "docs/other.md", "tests/fixtures/private.sqlite3", "tests/.pytest_cache/cached.py",
        "paper_evidence_rag/app.py", "quadratic_reference/app.py",
    }
    for name in allowed | excluded:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content for " + name, encoding="utf-8")
    before = {name: (project / name).read_bytes() for name in excluded}
    output = tmp_path / "handoff.zip"

    count = package_api().package_project(project, output)

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert count == len(names)
        assert {"shu-insight/" + name for name in allowed} <= names
        assert {"shu-insight/" + name for name in excluded}.isdisjoint(names)
        assert "shu-insight/app.py" in names
        assert "shu-insight/data/official/preload-2026-10-25.json" in names
        assert archive.read("shu-insight/docs/组员协作与运行.md").decode() == "组员运行说明"
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
    assert before == {name: (project / name).read_bytes() for name in excluded}


def test_package_is_reproducible_and_does_not_overwrite(project, tmp_path):
    api = package_api()
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    api.package_project(project, first)
    api.package_project(project, second)
    assert first.read_bytes() == second.read_bytes()
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        api.package_project(project, first)
    assert first.read_bytes() == original


def test_missing_required_source_does_not_leave_an_archive(project, tmp_path):
    (project / "pyproject.toml").unlink()
    output = tmp_path / "incomplete.zip"
    with pytest.raises(ValueError, match="pyproject.toml"):
        package_api().package_project(project, output)
    assert not output.exists()


def test_linked_source_outside_project_is_refused(project, tmp_path):
    outside = tmp_path / "private.py"
    outside.write_text("SECRET='do not share'", encoding="utf-8")
    link = project / "src/shu_insight/linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("this Windows account cannot create symbolic links")
    output = tmp_path / "unsafe.zip"
    with pytest.raises(ValueError, match="linked|outside"):
        package_api().package_project(project, output)
    assert not output.exists()


def test_windows_reparse_directory_is_refused_before_output(project, tmp_path, monkeypatch):
    original_lstat = Path.lstat

    def reparse_stat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == project / "src":
            return SimpleNamespace(st_mode=info.st_mode,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info

    monkeypatch.setattr(Path, "lstat", reparse_stat)
    output = tmp_path / "unsafe.zip"
    with pytest.raises(ValueError, match="linked"):
        package_api().package_project(project, output)
    assert not output.exists()
