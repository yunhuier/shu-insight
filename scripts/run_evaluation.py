from __future__ import annotations

import argparse
import os
from pathlib import Path

from shu_insight.evaluation.dataset import load_frozen_dataset
from shu_insight.evaluation.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen SHU Insight experiment matrix.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mini-fixture", action="store_true")
    configured_model = os.getenv("SHU_NOTICE_SEMANTIC_MODEL_PATH", "").strip()
    parser.add_argument(
        "--semantic-model",
        type=Path,
        default=Path(configured_model) if configured_model else None,
        help="optional already-cached local sentence-transformer directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_frozen_dataset(
        args.dataset,
        allow_mini_fixture=args.allow_mini_fixture,
    )
    report = ExperimentRunner(
        dataset,
        semantic_model_path=args.semantic_model,
    ).run()
    report.write_json(args.output)
    print(
        f"wrote {len(report.experiments)} experiments for {report.row_count} rows "
        f"({report.corpus_kind}; formal_metrics_available={report.formal_metrics_available}) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
