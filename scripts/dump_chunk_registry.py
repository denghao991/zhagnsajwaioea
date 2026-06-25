"""生成 chunk_registry.json — chunk_id → 全文 映射表。

用于离线分析时查日志中的 chunk_id 对应什么内容。

用法：
    # 从 data/eval/md 生成
    python scripts/dump_chunk_registry.py

    # 从自定义目录生成
    python scripts/dump_chunk_registry.py --doc-dir data/eval/md

    # 指定输出路径
    python scripts/dump_chunk_registry.py -o data/json/chunk_registry.json
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tiny_rag.ingestion.chunker import MarkdownChunker


def build_registry(doc_dir: str) -> dict[str, str]:
    """Chunk all .md files and return {chunk_id: full_text}."""
    chunker = MarkdownChunker()
    registry: dict[str, str] = {}

    md_files = sorted(Path(doc_dir).glob("*.md"))
    if not md_files:
        print(f"  ERROR: 在 {doc_dir} 中未找到 .md 文件")
        return registry

    for fpath in md_files:
        text = fpath.read_text(encoding="utf-8")
        results = chunker.chunk_text(text)
        for i, c in enumerate(results):
            chunk_id = f"{fpath.stem}#{i}"
            registry[chunk_id] = c.text
            print(f"  + {chunk_id}  ({c.token_count} tokens)")

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 chunk_registry.json")
    parser.add_argument("--doc-dir", default="data/eval/md",
                        help="文档目录，默认 data/eval/md")
    parser.add_argument("-o", "--output", default="data/json/chunk_registry.json",
                        help="输出路径，默认 data/json/chunk_registry.json")
    args = parser.parse_args()

    doc_dir = str((_project_root / args.doc_dir).resolve())
    out_path = (_project_root / args.output).resolve()

    print(f"从 {doc_dir} 分块...")
    registry = build_registry(doc_dir)

    if not registry:
        print("ERROR: 未生成任何 chunk")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\n  {len(registry)} chunks → {out_path}")


if __name__ == "__main__":
    main()
