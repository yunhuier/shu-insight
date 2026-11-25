"""Export authentic local official evidence without contacting any website."""

from __future__ import annotations

import argparse
from pathlib import Path

from shu_insight.ingest.preload import (
    export_preload,
    reviewed_attachment_exclusion,
    reviewed_computer_article,
)
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewed-computer-article", type=Path,
                        help="optional exact individually reviewed saved HTML; never downloaded")
    parser.add_argument("--reviewed-attachment-exclusion", type=Path,
                        help="exact reviewed saved attachment-only HTML; never downloaded")
    args = parser.parse_args()
    registry = load_source_registry(ROOT / "config/sources.toml")
    reviewed = ([reviewed_computer_article(args.reviewed_computer_article, registry)]
                if args.reviewed_computer_article else [])
    exclusions = ([reviewed_attachment_exclusion(args.reviewed_attachment_exclusion, registry)]
                  if args.reviewed_attachment_exclusion else [])
    report = export_preload(args.database, args.manifest, registry, reviewed_entries=reviewed,
                            reviewed_exclusions=exclusions)
    print(f"exported={report.imported_count} rejected={report.rejected_count}")


if __name__ == "__main__":
    main()
