"""Evidence collection helpers for heterogeneous Agent/RAG outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from .schemas import EvidenceItem, EvidenceType


EVIDENCE_KEYS = {
    "sources",
    "documents",
    "retrieved_docs",
    "context",
    "tool_outputs",
    "sql_result",
    "sql_rows",
    "cypher_records",
    "cyphers",
    "evidence",
    "rows",
    "records",
}


class GroundingCollector:
    """Normalize evidence from sources, docs, rows, cypher records, and tool outputs."""

    def __init__(self, *, max_items: int = 80, max_content_chars: int = 1200) -> None:
        self.max_items = max_items
        self.max_content_chars = max_content_chars

    def collect(self, result: Dict[str, Any], route: Optional[str] = None) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        seen: Set[int] = set()

        def add_item(value: Any, source_type: EvidenceType = "unknown", source_name: Optional[str] = None) -> None:
            if len(evidence) >= self.max_items:
                return
            item = self._to_evidence(value, source_type=source_type, source_name=source_name, route=route)
            if item and item.content.strip():
                key = (item.source_type, item.source_name, item.content[:200])
                if key not in dedupe:
                    dedupe.add(key)
                    evidence.append(item)

        dedupe: Set[tuple[str, Optional[str], str]] = set()

        # Fast path for common top-level payloads.
        for key in ("sources", "documents"):
            if key in result:
                self._collect_value(result[key], key, add_item)

        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key in ("sources", "agent_state"):
                if key in metadata:
                    self._collect_value(metadata[key], key, add_item, seen)

        if "agent_state" in result:
            self._collect_value(result["agent_state"], "agent_state", add_item, seen)

        self._collect_value(result, "root", add_item, seen)
        return evidence

    def _collect_value(
        self,
        value: Any,
        key_hint: str,
        add_item,
        seen: Optional[Set[int]] = None,
        *,
        depth: int = 0,
    ) -> None:
        if depth > 7:
            return
        seen = seen if seen is not None else set()
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if hasattr(value, "additional_kwargs"):
            extra = getattr(value, "additional_kwargs", {}) or {}
            for key in ("sources", "tool_outputs"):
                if key in extra:
                    self._collect_value(extra[key], key, add_item, seen, depth=depth + 1)

        if isinstance(value, list):
            for item in value:
                self._collect_value(item, key_hint, add_item, seen, depth=depth + 1)
            return

        if isinstance(value, tuple):
            for item in value:
                self._collect_value(item, key_hint, add_item, seen, depth=depth + 1)
            return

        if isinstance(value, dict):
            if self._looks_like_evidence_dict(value, key_hint):
                add_item(value, self._source_type_for_key(key_hint), self._source_name(value))

            for key, child in value.items():
                if key in EVIDENCE_KEYS:
                    self._collect_value(child, key, add_item, seen, depth=depth + 1)
                elif key == "safety" and isinstance(child, dict):
                    if "evidence" in child:
                        self._collect_value(child["evidence"], "evidence", add_item, seen, depth=depth + 1)
                elif key in {"metadata", "additional_kwargs"} and isinstance(child, dict):
                    for nested_key in ("sources", "tool_outputs"):
                        if nested_key in child:
                            self._collect_value(child[nested_key], nested_key, add_item, seen, depth=depth + 1)
            return

        if key_hint in EVIDENCE_KEYS and isinstance(value, (str, int, float)):
            add_item(str(value), self._source_type_for_key(key_hint), key_hint)

    def _to_evidence(
        self,
        value: Any,
        *,
        source_type: EvidenceType,
        source_name: Optional[str],
        route: Optional[str],
    ) -> Optional[EvidenceItem]:
        metadata: Dict[str, Any] = {"route": route} if route else {}
        score: Optional[float] = None

        if isinstance(value, EvidenceItem):
            return value

        if isinstance(value, dict):
            if value.get("source_type"):
                source_type = value.get("source_type")  # type: ignore[assignment]
            metadata.update({k: self._safe_json(v) for k, v in value.items() if k not in {"content", "text", "answer"}})
            content = (
                value.get("content")
                or value.get("text")
                or value.get("answer")
                or value.get("summary")
                or value.get("result")
                or value.get("source")
                or value.get("document_id")
                or self._safe_json(value)
            )
            source_name = source_name or self._source_name(value)
            raw_score = value.get("score") or value.get("similarity") or value.get("rerank_score")
            if raw_score is not None:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
        else:
            content = str(value)

        text = str(content or "").strip()
        if not text:
            return None

        return EvidenceItem(
            source_type=source_type,
            content=text[: self.max_content_chars],
            source_name=source_name,
            score=score,
            metadata=metadata,
        )

    @staticmethod
    def _looks_like_evidence_dict(value: Dict[str, Any], key_hint: str) -> bool:
        if key_hint in EVIDENCE_KEYS:
            return True
        evidence_fields = {
            "content",
            "text",
            "answer",
            "summary",
            "result",
            "rows",
            "records",
            "document_id",
            "source",
        }
        return bool(evidence_fields.intersection(value.keys()))

    @staticmethod
    def _source_type_for_key(key: str) -> EvidenceType:
        mapping: Dict[str, EvidenceType] = {
            "sources": "document",
            "documents": "document",
            "retrieved_docs": "kb_chunk",
            "context": "kb_chunk",
            "tool_outputs": "tool_output",
            "sql_result": "sql_row",
            "sql_rows": "sql_row",
            "rows": "sql_row",
            "cypher_records": "cypher_record",
            "cyphers": "cypher_record",
            "records": "cypher_record",
        }
        return mapping.get(key, "unknown")

    @staticmethod
    def _source_name(value: Dict[str, Any]) -> Optional[str]:
        for key in ("source", "source_table", "document_id", "id", "name", "title", "tool"):
            if value.get(key):
                return str(value[key])
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            for key in ("source", "source_table", "document_id", "id", "name", "title"):
                if metadata.get(key):
                    return str(metadata[key])
        return None

    @staticmethod
    def _safe_json(value: Any) -> Any:
        try:
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return str(value)
