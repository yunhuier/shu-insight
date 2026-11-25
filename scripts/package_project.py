"""Create a portable source ZIP from an explicit allowlist, without local state."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "app.py", "pyproject.toml", "README.md", ".gitignore", ".env.example", "run_local.ps1",
    "src/shu_insight/__init__.py", "config/sources.toml", "config/taxonomy.toml",
    "config/ranking.toml", "scripts/import_official_preload.py", "scripts/package_project.py",
    "docs/组员协作与运行.md", "data/official/preload-2026-10-25.json",
)
OPTIONAL_FILES = (
    "start.bat",
    "scripts/cache_semantic_model.py", "scripts/export_annotation_queue.py",
    "scripts/export_official_preload.py", "scripts/freeze_annotations.py",
    "scripts/import_official_snapshot.py", "scripts/run_evaluation.py", "scripts/seed_demo.py",
    "data/official/snapshot-2026-09-24.json", "data/demo/notices.json",
    "docs/上大知讯-新增功能图解.docx",
)
FIXTURE_EXTENSIONS = frozenset({".html", ".sql", ".jsonl"})


def _check_path(root: Path, relative: Path) -> Path:
    """Refuse links/junctions before reading any selected file or directory."""
    path = root
    for part in relative.parts:
        path /= part
        info = path.lstat()
        reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise ValueError(f"linked project path is not shareable: {relative.as_posix()}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"project path resolves outside root: {relative.as_posix()}")
    return path


def _tree_files(root: Path, folder: str) -> set[Path]:
    tree = root / folder
    if not tree.exists():
        return set()
    _check_path(root, Path(folder))
    selected = set()
    for current, directories, files in os.walk(tree, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if not name.startswith(".") and name != "__pycache__")
        current_path = Path(current)
        for name in directories:
            _check_path(root, (current_path / name).relative_to(root))
        for name in sorted(files):
            if name.startswith("."):
                continue
            relative = (current_path / name).relative_to(root)
            allowed = relative.suffix == ".py"
            if folder == "tests" and relative.parts[1:2] == ("fixtures",):
                allowed = relative.suffix in FIXTURE_EXTENSIONS
            if allowed:
                _check_path(root, relative)
                selected.add(relative)
    return selected


def package_project(root: Path, output: Path) -> int:
    """Write a deterministic ZIP; require an explicit output and never overwrite."""
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output.suffix.lower() != ".zip":
        raise ValueError("output filename must end in .zip")
    selected = set()
    for name in REQUIRED_FILES:
        relative = Path(name)
        try:
            path = _check_path(root, relative)
        except FileNotFoundError as exc:
            raise ValueError(f"required project file missing: {name}") from exc
        if not path.is_file():
            raise ValueError(f"required project file is not a regular file: {name}")
        selected.add(relative)
    for name in OPTIONAL_FILES:
        relative = Path(name)
        if (root / relative).exists():
            _check_path(root, relative)
            selected.add(relative)
    for path in sorted((root / "config").glob("*.toml")):
        relative = path.relative_to(root)
        _check_path(root, relative)
        selected.add(relative)
    selected |= _tree_files(root, "src") | _tree_files(root, "tests")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses an existing output symlink. Remove only
    # the incomplete file created by this invocation if ZIP writing fails.
    with output.open("xb") as destination:
        try:
            with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
                for relative in sorted(selected, key=lambda item: item.as_posix()):
                    path = _check_path(root, relative)
                    if not path.is_file():
                        raise ValueError(f"not a regular project file: {relative.as_posix()}")
                    info = ZipInfo("shu-insight/" + relative.as_posix(), (1980, 1, 1, 0, 0, 0))
                    info.compress_type = ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, path.read_bytes(), compresslevel=9)
        except BaseException:
            destination.close()
            output.unlink()
            raise
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new .zip path; never overwritten")
    args = parser.parse_args()
    try:
        count = package_project(ROOT, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot create project ZIP: {exc}\n")
    print(f"Created {args.output.absolute()} ({count} files)")


if __name__ == "__main__":
    main()
