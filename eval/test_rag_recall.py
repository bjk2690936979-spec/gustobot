"""Small offline RAG recall and context-size evaluation.

Run from the project root:
    python eval/test_rag_recall.py

This script uses an in-memory vector store so it is safe to run without Milvus,
OpenAI, or Docker. It measures the retrieval pipeline changes in
KnowledgeService: raw recall, query rewrite, multi-query merge, de-duplication,
and context compression.
"""
from __future__ import annotations

import asyncio
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The eval uses an in-memory store, but importing KnowledgeService imports the
# production Milvus wrapper. Provide a tiny stub when pymilvus is not installed
# locally so the offline script remains runnable without Docker.
try:  # pragma: no cover - depends on local environment
    import pymilvus  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local eval convenience
    pymilvus_stub = types.ModuleType("pymilvus")

    class _Dummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _Utility:
        @staticmethod
        def has_collection(name: str) -> bool:
            return False

    class _Connections:
        @staticmethod
        def connect(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def disconnect(*args: Any, **kwargs: Any) -> None:
            return None

    class _DataType:
        VARCHAR = "VARCHAR"
        FLOAT_VECTOR = "FLOAT_VECTOR"

    pymilvus_stub.connections = _Connections()
    pymilvus_stub.Collection = _Dummy
    pymilvus_stub.CollectionSchema = _Dummy
    pymilvus_stub.FieldSchema = _Dummy
    pymilvus_stub.DataType = _DataType
    pymilvus_stub.utility = _Utility()
    sys.modules["pymilvus"] = pymilvus_stub

try:  # pragma: no cover - depends on local environment
    import langchain_core.documents  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local eval convenience
    langchain_core_stub = types.ModuleType("langchain_core")
    documents_stub = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_stub.Document = Document
    langchain_core_stub.documents = documents_stub
    sys.modules["langchain_core"] = langchain_core_stub
    sys.modules["langchain_core.documents"] = documents_stub

try:  # pragma: no cover - depends on local environment
    import langchain_text_splitters  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local eval convenience
    splitters_stub = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def split_documents(self, documents: List[Any]) -> List[Any]:
            return documents

    splitters_stub.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = splitters_stub

try:  # pragma: no cover - depends on local environment
    import neo4j  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local eval convenience
    neo4j_stub = types.ModuleType("neo4j")

    class GraphDatabase:
        @staticmethod
        def driver(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Neo4j is not available in offline eval")

    class Result:
        pass

    neo4j_stub.GraphDatabase = GraphDatabase
    neo4j_stub.Result = Result
    neo4j_graph_stub = types.ModuleType("neo4j.graph")
    neo4j_graph_stub.Graph = type("Graph", (), {})
    neo4j_graph_stub.Node = type("Node", (), {})
    neo4j_graph_stub.Relationship = type("Relationship", (), {})
    sys.modules["neo4j"] = neo4j_stub
    sys.modules["neo4j.graph"] = neo4j_graph_stub

from gustobot.config import settings
from gustobot.infrastructure.knowledge import KnowledgeService
from loguru import logger as loguru_logger

loguru_logger.remove()


class FakeEmbedder:
    def embed_query(self, text: str) -> List[Any]:
        return [text]


class DisabledReranker:
    enabled = False

    async def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return documents[:top_k]


class InMemoryVectorStore:
    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents

    def query_by_name(self, name_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (name_query or "").strip()
        if not query:
            return []
        matches = []
        for doc in self.documents:
            name = str(doc.get("metadata", {}).get("name") or "")
            if query == name or query in name or name in query:
                item = dict(doc)
                item["score"] = max(float(item.get("score") or 0.0), 1.0)
                matches.append(item)
        return matches[:limit]

    def search(self, query_embedding: List[Any], top_k: int = 10, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        query = str(query_embedding[0] if query_embedding else "")
        ranked = []
        for doc in self.documents:
            score = self._score(query, doc)
            if score <= 0:
                continue
            item = dict(doc)
            item["metadata"] = dict(doc.get("metadata") or {})
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _score(query: str, doc: Dict[str, Any]) -> float:
        name = str(doc.get("metadata", {}).get("name") or "")
        content = str(doc.get("content") or "")
        haystack = f"{name} {content}".lower()
        query_text = query.lower()
        score = 0.0
        if name and name.lower() in query_text:
            score += 5.0
        for token in _tokens(query_text):
            if token and token in haystack:
                score += 1.0
        return score


def _tokens(text: str) -> List[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    parts = [part for part in text.split() if part]
    # Add short Chinese n-grams so queries without spaces can still match.
    compact = "".join(parts)
    grams = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    grams.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
    return parts + grams


def _doc(doc_id: str, name: str, category: str, content: str) -> Dict[str, Any]:
    long_tail = " 烹饪背景与资料补充。" * 220
    return {
        "id": doc_id,
        "content": f"菜名：{name}\n类型：{category}\n{content}{long_tail}",
        "score": 0.0,
        "metadata": {
            "recipe_id": doc_id,
            "name": name,
            "category": category,
            "source": "offline_eval",
        },
    }


DOCUMENTS = [
    _doc("recipe_001", "宫保鸡丁", "川菜", "特色：糊辣荔枝味。历史：宫保二字与丁宝桢相关。做法：鸡丁、花生、干辣椒炒制。"),
    _doc("recipe_002", "香肠炒菜干", "热菜", "特色：酱香浓郁，菜干吸收香肠油脂。食材：香肠、菜干、豆豉、蒜。步骤：先爆香再翻炒。"),
    _doc("recipe_003", "红烧豆腐", "家常菜", "做法：豆腐煎香后加入调味汁焖煮。特点：咸鲜软嫩。"),
    _doc("recipe_004", "九转大肠", "鲁菜", "特色：酸甜咸鲜辣五味俱全，是经典鲁菜代表。"),
    _doc("recipe_005", "葱烧海参", "鲁菜", "特色：葱香浓郁，海参软糯，是鲁菜高汤与火候代表。"),
    _doc("recipe_006", "麻婆豆腐", "川菜", "特点：麻、辣、烫、香、酥、嫩。做法：豆腐与牛肉末、豆瓣酱烧制。"),
    _doc("recipe_007", "糖醋鲤鱼", "鲁菜", "特色：外酥里嫩，酸甜口，造型讲究。"),
    _doc("recipe_008", "西红柿炒鸡蛋", "家常菜", "做法：鸡蛋先炒成块，再和西红柿一起翻炒。"),
    _doc("recipe_009", "马铃薯炖牛肉", "家常菜", "做法：马铃薯切块，与牛肉一起小火炖煮至软烂。"),
]


CASES = [
    {"query": "「宫保鸡丁」为什么叫宫保", "expected_id": "recipe_001"},
    {"query": "香肠炒菜干有什么特点", "expected_id": "recipe_002"},
    {"query": "红烧豆腐怎么做", "expected_id": "recipe_003"},
    {"query": "请推荐经典鲁菜并说明特色", "expected_keyword": "鲁菜"},
    {"query": "麻婆豆腐有哪些特点", "expected_id": "recipe_006"},
    {"query": "糖醋鲤鱼是什么口味", "expected_id": "recipe_007"},
    {"query": "番茄炒蛋怎么做", "expected_id": "recipe_008"},
    {"query": "葱烧海参体现了什么风味", "expected_id": "recipe_005"},
    {"query": "土豆炖牛肉怎么做", "expected_id": "recipe_009"},
]


@contextmanager
def temporary_settings(**overrides: Any):
    original = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def _match_rank(results: List[Dict[str, Any]], case: Dict[str, str]) -> Optional[int]:
    expected_id = case.get("expected_id")
    expected_keyword = case.get("expected_keyword")
    for idx, doc in enumerate(results):
        metadata = doc.get("metadata") or {}
        ids = {
            str(doc.get("id") or ""),
            str(metadata.get("id") or ""),
            str(metadata.get("recipe_id") or ""),
            str(metadata.get("chunk_id") or ""),
        }
        text = f"{metadata.get('name', '')} {metadata.get('category', '')} {doc.get('content', '')}"
        if expected_id and expected_id in ids:
            return idx + 1
        if expected_keyword and expected_keyword in text:
            return idx + 1
    return None


async def run_eval(label: str, **config: Any) -> Dict[str, Any]:
    service = KnowledgeService(
        vector_store=InMemoryVectorStore(DOCUMENTS),
        embedder=FakeEmbedder(),
        reranker=DisabledReranker(),
    )

    ranks: List[Optional[int]] = []
    diagnostics: List[Dict[str, Any]] = []
    bad_cases: List[Dict[str, Any]] = []
    with temporary_settings(**config):
        for case in CASES:
            results = await service.search(
                case["query"],
                raw_top_k=settings.RAG_RAW_TOP_K,
                final_top_k=settings.RAG_FINAL_TOP_K,
                max_chunk_chars=settings.RAG_MAX_CHUNK_CHARS,
                similarity_threshold=0.0,
            )
            rank = _match_rank(results, case)
            ranks.append(rank)
            diagnostics.append(dict(service.last_search_diagnostics))
            if rank is None:
                bad_cases.append(
                    {
                        "query": case["query"],
                        "expected": case.get("expected_id") or case.get("expected_keyword"),
                        "actual": [
                            (doc.get("metadata") or {}).get("name") or doc.get("id")
                            for doc in results[:5]
                        ],
                    }
                )

    def recall_at(k: int) -> float:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        return hits / len(ranks)

    before = sum(item.get("context_before_chars", 0) for item in diagnostics)
    after = sum(item.get("context_after_chars", 0) for item in diagnostics)
    reduction = 1 - (after / before) if before else 0.0

    return {
        "label": label,
        "total": len(CASES),
        "recall": {k: recall_at(k) for k in (1, 3, 5, 10)},
        "raw_chunks": sum(item.get("raw_count", 0) for item in diagnostics),
        "dedup_chunks": sum(item.get("dedup_count", 0) for item in diagnostics),
        "final_chunks": sum(item.get("final_count", 0) for item in diagnostics),
        "before_chars": before,
        "after_chars": after,
        "reduction": reduction,
        "bad_cases": bad_cases,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n[{report['label']}]")
    print(f"Total cases: {report['total']}")
    for k in (1, 3, 5, 10):
        print(f"Recall@{k}: {report['recall'][k] * 100:.2f}%")
    print(f"Raw chunks retrieved: {report['raw_chunks']}")
    print(f"Chunks after dedup: {report['dedup_chunks']}")
    print(f"Final chunks for LLM: {report['final_chunks']}")
    print(f"Context before compression: {report['before_chars']} chars")
    print(f"Context after compression: {report['after_chars']} chars")
    print(f"Context reduction: {report['reduction'] * 100:.2f}%")
    print("Bad cases:")
    if not report["bad_cases"]:
        print("  None")
    for item in report["bad_cases"]:
        print(f"  query={item['query']} expected={item['expected']} actual={item['actual']}")


async def main() -> None:
    baseline = await run_eval(
        "baseline",
        RAG_RAW_TOP_K=5,
        RAG_FINAL_TOP_K=5,
        RAG_MAX_CHUNK_CHARS=4000,
        ENABLE_QUERY_REWRITE=False,
        ENABLE_MULTI_QUERY=False,
        ENABLE_CONTEXT_COMPRESSION=False,
        RERANK_ENABLED=False,
    )
    optimized = await run_eval(
        "optimized",
        RAG_RAW_TOP_K=20,
        RAG_FINAL_TOP_K=5,
        RAG_MAX_CHUNK_CHARS=800,
        ENABLE_QUERY_REWRITE=True,
        ENABLE_MULTI_QUERY=True,
        ENABLE_CONTEXT_COMPRESSION=True,
        RERANK_ENABLED=False,
    )

    print_report(baseline)
    print_report(optimized)


if __name__ == "__main__":
    asyncio.run(main())
