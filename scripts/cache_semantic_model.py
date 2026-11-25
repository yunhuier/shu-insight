from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "BAAI/bge-small-zh-v1.5"
APPROXIMATE_DISK_REQUIREMENT = "约 500 MB（模型与下载缓存）"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly download the optional local semantic relevance model."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/models/bge-small-zh-v1.5"),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the displayed destination and approximate disk requirement.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination = args.destination.resolve()
    print(f"模型：{MODEL_ID}")
    print(f"目标目录：{destination}")
    print(f"预计空间：{APPROXIMATE_DISK_REQUIREMENT}")
    if not args.yes:
        print("未下载。确认后请重新运行并添加 --yes。")
        return 2

    from sentence_transformers import SentenceTransformer

    destination.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID)
    model.save(str(destination))
    print("模型已缓存；应用启动时只会从该本地目录加载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
