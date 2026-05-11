#!/usr/bin/env python3
"""
Fast direct importer for data/recipe.json into the Milvus-backed KB.

This script runs inside the backend environment and calls KnowledgeService
directly, avoiding the HTTP API timeout that can happen with large batches.

Example:
  python scripts/import_recipes_direct.py --file data/recipe.json --limit 5000 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gustobot.infrastructure.knowledge import KnowledgeService


T = TypeVar("T")


def _chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalize_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{k} {v}".strip() for k, v in value.items() if str(k).strip()]
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                name = str(item[0]).strip() if item else ""
                amount = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
                if name:
                    items.append(f"{name} {amount}".strip())
            else:
                text = str(item).strip()
                if text:
                    items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _split_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Bundled data commonly stores steps as: 1:xxx2:yyy3:zzz
    import re

    parts = [part.strip() for part in re.split(r"\s*\d+\s*[:：]\s*", text) if part.strip()]
    if len(parts) > 1:
        return parts
    return [part.strip() for part in re.split(r"[。；;]\s*", text) if part.strip()]


def _first(entry: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value not in (None, ""):
            return value
    return None


def _recipe_to_text(name: str, entry: Dict[str, Any]) -> str:
    category = _first(entry, "category", "类型", "分类")
    time_cost = _first(entry, "time", "cook_time", "耗时")
    taste = _first(entry, "taste", "口味")
    craft = _first(entry, "craft", "工艺")
    description = _first(entry, "description", "简介", "描述")

    ingredients = []
    ingredients.extend(_normalize_items(_first(entry, "main_ingredients", "主食材")))
    ingredients.extend(_normalize_items(_first(entry, "ingredients", "ingredient_list", "辅料")))

    steps = _split_steps(_first(entry, "steps", "做法"))
    tips = _first(entry, "tips", "小贴士")

    parts: List[str] = [f"菜名：{name}"]
    if category:
        parts.append(f"类型：{category}")
    if time_cost:
        parts.append(f"耗时：{time_cost}")
    if taste:
        parts.append(f"口味：{taste}")
    if craft:
        parts.append(f"工艺：{craft}")
    if description:
        parts.append(f"简介：{description}")
    if ingredients:
        parts.append("食材：" + "、".join(ingredients))
    for idx, step in enumerate(steps, start=1):
        parts.append(f"步骤{idx}：{step}")
    if tips:
        parts.append(f"小贴士：{tips}")
    return "\n".join(parts)


def _limit_text(text: str, *, max_lines: Optional[int], max_chars: Optional[int]) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if max_lines is not None and max_lines > 0:
        text = "\n".join(lines[:max_lines])
    else:
        text = "\n".join(lines)
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _apply_content_mode(
    text: str,
    *,
    mode: str,
    max_lines: Optional[int],
    max_chars: Optional[int],
) -> str:
    if mode == "full":
        return _limit_text(text, max_lines=max_lines, max_chars=max_chars)
    if mode == "compact":
        return _limit_text(
            text,
            max_lines=max_lines if max_lines is not None else 10,
            max_chars=max_chars if max_chars is not None else 700,
        )
    if mode == "minimal":
        return _limit_text(
            text,
            max_lines=max_lines if max_lines is not None else 5,
            max_chars=max_chars if max_chars is not None else 320,
        )
    raise ValueError(f"Unsupported content mode: {mode}")


def _iter_recipe_entries(raw: Any, *, limit: Optional[int] = None) -> Iterator[tuple[int, str, Dict[str, Any]]]:
    yield from _iter_recipe_entries_from(raw, start_index=0, limit=limit)


def _iter_recipe_entries_from(
    raw: Any,
    *,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> Iterator[tuple[int, str, Dict[str, Any]]]:
    emitted = 0
    if isinstance(raw, dict):
        iterable: Iterable[tuple[str, Any]] = raw.items()
    elif isinstance(raw, list):
        iterable = ((str(idx), item) for idx, item in enumerate(raw))
    else:
        raise TypeError(f"Unsupported JSON root type: {type(raw)!r}")

    for idx, (name_or_idx, entry) in enumerate(iterable):
        if idx < start_index:
            continue
        if limit is not None and emitted >= limit:
            return
        if not isinstance(entry, dict):
            continue
        name = (
            str(_first(entry, "name", "title", "菜名") or name_or_idx).strip()
            or f"recipe_{idx}"
        )
        emitted += 1
        yield idx, name, entry


def _flush_batch(service: KnowledgeService, docs: List[Any], *, embedding_batch_size: int) -> int:
    if not docs:
        return 0
    embeddings: List[List[float]] = []
    texts = [doc.page_content for doc in docs]
    for text_batch in _chunked(texts, embedding_batch_size):
        embeddings.extend(service.embedder.embed_documents(list(text_batch)))
    result = service._store_documents(docs, embeddings)  # noqa: SLF001 - intentional import utility
    return int(result.get("add_count", 0))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Directly import recipe JSON into Milvus KB")
    parser.add_argument("--file", required=True, help="Path to recipe JSON file")
    parser.add_argument("--start-index", type=int, default=0, help="Skip recipes before this JSON index")
    parser.add_argument("--limit", type=int, default=None, help="Only import first N recipes")
    parser.add_argument("--batch-size", type=int, default=128, help="Document chunks per Milvus store batch")
    parser.add_argument("--embedding-batch-size", type=int, default=10, help="Texts per embedding API request")
    parser.add_argument("--content-mode", choices=["full", "compact", "minimal"], default="full")
    parser.add_argument("--max-content-chars", type=int, default=None, help="Truncate each recipe text before embedding")
    parser.add_argument("--max-content-lines", type=int, default=None, help="Keep only the first N lines before embedding")
    parser.add_argument("--no-split", action="store_true", help="Store one vector per recipe instead of chunking")
    parser.add_argument("--clear-first", action="store_true", help="Clear Milvus collection before importing")
    parser.add_argument("--embedding-timeout", type=float, default=180.0, help="Embedding API timeout seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.embedding_batch_size <= 0:
        parser.error("--embedding-batch-size must be greater than 0")
    if args.start_index < 0:
        parser.error("--start-index must be greater than or equal to 0")
    if args.max_content_chars is not None and args.max_content_chars <= 0:
        parser.error("--max-content-chars must be greater than 0")
    if args.max_content_lines is not None and args.max_content_lines <= 0:
        parser.error("--max-content-lines must be greater than 0")

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"File not found: {json_path}", file=sys.stderr)
        return 2

    with json_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    service = KnowledgeService()
    service.embedder.request_timeout = args.embedding_timeout

    if args.clear_first:
        print("Clearing Milvus collection before import...")
        service.vector_store.clear_collection()

    started = time.time()
    pending_docs: List[Any] = []
    total_recipes = 0
    total_chunks = 0
    total_added = 0

    try:
        for idx, name, entry in _iter_recipe_entries_from(raw, start_index=args.start_index, limit=args.limit):
            text = _apply_content_mode(
                _recipe_to_text(name, entry),
                mode=args.content_mode,
                max_lines=args.max_content_lines,
                max_chars=args.max_content_chars,
            )
            metadata = {
                "recipe_id": f"recipe_{idx}",
                "name": name,
                "title": name,
                "category": str(_first(entry, "category", "类型", "分类") or ""),
                "source": json_path.name,
                "import_mode": "direct_recipe_json",
                "content_mode": args.content_mode,
            }
            chunks = (
                [Document(page_content=text, metadata=metadata)]
                if args.no_split
                else service._split_into_documents(text, metadata)  # noqa: SLF001
            )
            pending_docs.extend(chunks)
            total_recipes += 1
            total_chunks += len(chunks)

            if len(pending_docs) >= args.batch_size:
                added = _flush_batch(
                    service,
                    pending_docs,
                    embedding_batch_size=args.embedding_batch_size,
                )
                total_added += added
                elapsed = max(time.time() - started, 0.001)
                print(
                    f"recipes={total_recipes} chunks={total_chunks} "
                    f"added={total_added} rate={total_recipes / elapsed:.1f} recipes/s",
                    flush=True,
                )
                pending_docs.clear()
                if args.sleep > 0:
                    time.sleep(args.sleep)

        if pending_docs:
            total_added += _flush_batch(
                service,
                pending_docs,
                embedding_batch_size=args.embedding_batch_size,
            )
            pending_docs.clear()

        elapsed = max(time.time() - started, 0.001)
        print(
            f"Done. recipes={total_recipes} chunks={total_chunks} "
            f"added={total_added} elapsed={elapsed:.1f}s rate={total_recipes / elapsed:.1f} recipes/s"
        )
        return 0
    finally:
        service.vector_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
