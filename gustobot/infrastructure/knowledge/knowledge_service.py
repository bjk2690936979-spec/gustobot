"""Knowledge base service implemented with LangChain primitives."""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from gustobot.config import settings
from .embeddings import OpenAICompatibleEmbeddings
from .vector_store import VectorStore
from .reranker import Reranker


class KnowledgeService:
    """Facade for ingesting documents and performing similarity search."""

    def __init__(
        self,
        *,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.KB_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.KB_CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " "],
            length_function=len,  # 使用字符长度而不是 tiktoken
        )

        if embedder is None:
            embedding_api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
            embedder = OpenAICompatibleEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=embedding_api_key,
                base_url=settings.EMBEDDING_BASE_URL,
                dimension=settings.EMBEDDING_DIMENSION,
            )
        self.embedder = embedder

        self.vector_store = vector_store or VectorStore(
            collection_name=settings.MILVUS_COLLECTION,
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            dimension=settings.EMBEDDING_DIMENSION,
            index_type=settings.MILVUS_INDEX_TYPE,
            metric_type=settings.MILVUS_METRIC_TYPE,
        )

        self.reranker = reranker or Reranker()
        self.last_search_diagnostics: Dict[str, Any] = {}

        logger.info(
            "KnowledgeService initialised (chunk_size=%s, chunk_overlap=%s)",
            self.chunk_size,
            self.chunk_overlap,
        )

    async def ingest_text(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"add_count": 0, "ids": []}

        documents = await asyncio.to_thread(self._split_into_documents, text, metadata or {})
        if not documents:
            return {"add_count": 0, "ids": []}

        embeddings = await asyncio.to_thread(
            self.embedder.embed_documents,
            [doc.page_content for doc in documents],
        )

        result = await asyncio.to_thread(self._store_documents, documents, embeddings)
        return result

    async def add_document(
        self,
        *,
        doc_id: Optional[str],
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        meta = metadata.copy() if metadata else {}
        meta.setdefault("title", title)
        meta.setdefault("name", title)
        if doc_id:
            meta.setdefault("recipe_id", doc_id)
        result = await self.ingest_text(content, metadata=meta)
        return result.get("add_count", 0) > 0

    async def add_recipe(self, recipe_id: str, recipe_data: Dict[str, Any]) -> bool:
        document = self._format_recipe_document(recipe_data)
        metadata = {
            "recipe_id": recipe_id,
            "name": recipe_data.get("name", ""),
            "category": recipe_data.get("category", ""),
            "difficulty": recipe_data.get("difficulty", ""),
        }
        result = await self.ingest_text(document, metadata=metadata)
        return result.get("add_count", 0) > 0

    async def add_recipes_batch(self, recipes: List[Dict[str, Any]]) -> Dict[str, int]:
        success_count = 0
        error_count = 0
        for recipe in recipes:
            recipe_id = recipe.get("id") or recipe.get("recipe_id") or str(uuid4())
            if await self.add_recipe(recipe_id, recipe):
                success_count += 1
            else:
                error_count += 1
        return {"success": success_count, "error": error_count, "total": len(recipes)}

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        raw_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        max_chunk_chars: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_expr: Optional[str] = None,
        filter_by_similarity: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            self.last_search_diagnostics = {
                "query": query,
                "queries": [],
                "raw_count": 0,
                "dedup_count": 0,
                "final_count": 0,
                "context_before_chars": 0,
                "context_after_chars": 0,
                "context_reduction_ratio": 0.0,
            }
            return []

        final_k = int(final_top_k or top_k or settings.RAG_FINAL_TOP_K or settings.KB_TOP_K)
        final_k = max(1, final_k)
        raw_k = int(raw_top_k or settings.RAG_RAW_TOP_K or final_k)
        raw_k = max(raw_k, final_k)
        chunk_char_limit = int(max_chunk_chars or settings.RAG_MAX_CHUNK_CHARS or 800)
        chunk_char_limit = max(120, chunk_char_limit)
        similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.KB_SIMILARITY_THRESHOLD
        )

        retrieval_queries = self._build_retrieval_queries(query)
        if not settings.ENABLE_MULTI_QUERY:
            retrieval_queries = [
                retrieval_queries[1]
                if settings.ENABLE_QUERY_REWRITE and len(retrieval_queries) > 1
                else retrieval_queries[0]
            ]

        candidates: List[Dict[str, Any]] = []
        name_candidates = self._extract_name_candidates(query)
        for rewritten in retrieval_queries:
            name_candidates.extend(self._extract_name_candidates(rewritten))

        for name_candidate in self._unique_strings(name_candidates):
            name_matches = await asyncio.to_thread(
                self.vector_store.query_by_name,
                name_candidate,
                raw_k,
            )
            for doc in name_matches:
                doc_copy = dict(doc)
                metadata = dict(doc_copy.get("metadata") or {})
                metadata.setdefault("match_type", "name")
                metadata.setdefault("source_query", name_candidate)
                doc_copy["metadata"] = metadata
                doc_copy.setdefault("score", 1.0)
                candidates.append(doc_copy)

        for retrieval_query in retrieval_queries:
            embedding = await asyncio.to_thread(self.embedder.embed_query, retrieval_query)
            results = await asyncio.to_thread(
                self.vector_store.search,
                embedding,
                raw_k,
                filter_expr,
            )
            for doc in results:
                doc_copy = dict(doc)
                metadata = dict(doc_copy.get("metadata") or {})
                metadata.setdefault("source_query", retrieval_query)
                metadata.setdefault("match_type", "vector")
                doc_copy["metadata"] = metadata
                candidates.append(doc_copy)

        raw_count = len(candidates)
        if filter_by_similarity and similarity_threshold is not None:
            candidates = [r for r in candidates if r.get("score", 0.0) >= similarity_threshold]

        deduped = self._dedupe_results(candidates)
        deduped.sort(key=self._result_rank_key, reverse=True)
        context_before_chars = sum(
            len(str(doc.get("content") or doc.get("document") or "")) for doc in deduped
        )

        # 使用 reranker 精排
        if deduped and self.reranker.enabled:
            rerank_limit = max(final_k, min(raw_k, settings.RERANK_MAX_CANDIDATES))
            deduped = await self.reranker.rerank(query, deduped, rerank_limit)

        if self.reranker.enabled:
            deduped = [
                r
                for r in deduped
                if r.get("rerank_score", 0.0) >= settings.KB_RERANK_SCORE_THRESHOLD
            ]
        elif not filter_by_similarity and similarity_threshold is not None:
            deduped = [r for r in deduped if r.get("score", 0.0) >= similarity_threshold]

        final_results = deduped[:final_k]
        if settings.ENABLE_CONTEXT_COMPRESSION:
            final_results = [
                self._compress_result(doc, max_chars=chunk_char_limit)
                for doc in final_results
            ]
        else:
            final_results = [self._sanitize_result_metadata(doc) for doc in final_results]

        context_after_chars = sum(
            len(str(doc.get("content") or doc.get("document") or "")) for doc in final_results
        )
        reduction_ratio = (
            1 - (context_after_chars / context_before_chars)
            if context_before_chars
            else 0.0
        )

        self.last_search_diagnostics = {
            "query": query,
            "queries": retrieval_queries,
            "raw_top_k": raw_k,
            "final_top_k": final_k,
            "raw_count": raw_count,
            "dedup_count": len(deduped),
            "final_count": len(final_results),
            "context_before_chars": context_before_chars,
            "context_after_chars": context_after_chars,
            "context_reduction_ratio": max(0.0, reduction_ratio),
        }
        logger.info(
            "RAG search query={} rewritten={} raw_count={} dedup_count={} final_count={} chars_before={} chars_after={}",
            query,
            retrieval_queries,
            raw_count,
            len(deduped),
            len(final_results),
            context_before_chars,
            context_after_chars,
        )

        return final_results

    def _build_retrieval_queries(self, query: str) -> List[str]:
        """Build low-cost query variants for higher recall without calling an LLM."""
        original = (query or "").strip()
        if not original:
            return []

        queries: List[str] = [original]
        if not settings.ENABLE_QUERY_REWRITE:
            return queries

        stripped = self._strip_query_noise(original)
        if stripped and stripped != original:
            queries.append(stripped)

        base_queries = [original]
        if stripped and stripped != original:
            base_queries.append(stripped)
        synonym_pairs = [
            ("番茄", "西红柿"),
            ("西红柿", "番茄"),
            ("土豆", "马铃薯"),
            ("马铃薯", "土豆"),
            ("炒蛋", "炒鸡蛋"),
            ("炒鸡蛋", "炒蛋"),
        ]
        for source, target in synonym_pairs:
            for base_query in base_queries:
                if source in base_query:
                    queries.append(base_query.replace(source, target))

        for candidate in self._extract_name_candidates(original):
            queries.append(candidate)
            if any(word in original for word in ("怎么做", "如何做", "做法", "步骤")):
                queries.append(f"{candidate} 做法 食材 步骤")
            if any(word in original for word in ("特点", "特色", "口味", "风味")):
                queries.append(f"{candidate} 特点 口味 风味")
            if any(word in original for word in ("历史", "典故", "由来", "为什么叫")):
                queries.append(f"{candidate} 历史 典故 由来")

        return self._unique_strings(queries)[:6]

    @staticmethod
    def _strip_query_noise(query: str) -> str:
        text = (query or "").strip()
        if not text:
            return ""
        replacements = [
            "请问",
            "请",
            "一下",
            "这个",
            "这道菜",
            "怎么做",
            "如何做",
            "做法",
            "是什么",
            "有什么特点",
            "有哪些特点",
            "特点",
            "特色",
            "介绍",
            "为什么叫",
            "为什么",
            "需要什么",
            "食材",
            "步骤",
            "历史",
            "典故",
            "由来",
        ]
        cleaned = text
        for phrase in replacements:
            cleaned = cleaned.replace(phrase, "")
        cleaned = re.sub(r"[\s，,。！？!?:：；;、]+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _unique_strings(values: List[str]) -> List[str]:
        seen: set[str] = set()
        unique: List[str] = []
        for value in values:
            normalized = (value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    @staticmethod
    def _result_rank_key(doc: Dict[str, Any]) -> float:
        rerank_score = doc.get("rerank_score")
        if isinstance(rerank_score, (int, float)):
            return float(rerank_score)
        score = doc.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        similarity = doc.get("similarity")
        if isinstance(similarity, (int, float)):
            return float(similarity)
        return 0.0

    def _dedupe_results(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_key: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            key = self._dedupe_key(doc)
            existing = best_by_key.get(key)
            if existing is None or self._result_rank_key(doc) > self._result_rank_key(existing):
                best_by_key[key] = doc
        return list(best_by_key.values())

    @staticmethod
    def _dedupe_key(doc: Dict[str, Any]) -> str:
        metadata = doc.get("metadata") or {}
        for key in ("recipe_id", "document_id", "source_id", "id", "chunk_id"):
            value = metadata.get(key) or doc.get(key)
            if value:
                if key == "chunk_id":
                    return f"chunk:{value}"
                return f"doc:{value}"
        content = str(doc.get("content") or doc.get("document") or "")
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
        return f"content:{digest}"

    def _compress_result(self, doc: Dict[str, Any], *, max_chars: int) -> Dict[str, Any]:
        result = self._sanitize_result_metadata(doc)
        content = str(result.get("content") or result.get("document") or "")
        if len(content) > max_chars:
            result["content"] = content[:max_chars].rstrip() + "..."
        else:
            result["content"] = content
        result.pop("document", None)
        return result

    @staticmethod
    def _sanitize_result_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(doc.get("metadata") or {})
        allowed_metadata = {
            "id",
            "recipe_id",
            "chunk_id",
            "source",
            "source_table",
            "url",
            "name",
            "title",
            "category",
            "difficulty",
            "match_type",
            "source_query",
        }
        clean_metadata = {
            key: value
            for key, value in metadata.items()
            if key in allowed_metadata and value not in (None, "")
        }
        clean_doc = {
            "id": doc.get("id") or metadata.get("id") or metadata.get("chunk_id"),
            "content": doc.get("content") or doc.get("document") or "",
            "score": doc.get("score"),
            "metadata": clean_metadata,
        }
        if doc.get("rerank_score") is not None:
            clean_doc["rerank_score"] = doc.get("rerank_score")
        if doc.get("similarity") is not None:
            clean_doc["similarity"] = doc.get("similarity")
        return clean_doc

    async def delete_recipe(self, recipe_id: str) -> bool:
        return await asyncio.to_thread(self.vector_store.delete_documents, [recipe_id])

    async def get_stats(self) -> Dict[str, Any]:
        def _stats() -> Dict[str, Any]:
            stats = self.vector_store.get_collection_stats()
            stats.update(
                {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "rag_raw_top_k": settings.RAG_RAW_TOP_K,
                    "rag_final_top_k": settings.RAG_FINAL_TOP_K,
                    "rag_max_chunk_chars": settings.RAG_MAX_CHUNK_CHARS,
                    "enable_query_rewrite": settings.ENABLE_QUERY_REWRITE,
                    "enable_multi_query": settings.ENABLE_MULTI_QUERY,
                    "enable_context_compression": settings.ENABLE_CONTEXT_COMPRESSION,
                }
            )
            return stats

        return await asyncio.to_thread(_stats)

    async def clear(self) -> bool:
        return await asyncio.to_thread(self.vector_store.clear_collection)

    async def close(self) -> None:
        await asyncio.to_thread(self.vector_store.close)

    def _split_into_documents(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        base_document = Document(page_content=text, metadata=metadata)
        chunks = self.splitter.split_documents([base_document])
        return chunks or [base_document]

    def _store_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        if not documents or not embeddings:
            return {"add_count": 0, "ids": [], "stored": False}

        ids: List[str] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for index, doc in enumerate(documents):
            metadata = dict(doc.metadata or {})
            base_id = metadata.get("recipe_id") or metadata.get("id") or metadata.get("source") or uuid4().hex
            chunk_id = f"{base_id}_{index}"
            metadata.setdefault("recipe_id", base_id)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault("name", metadata.get("name") or metadata.get("title") or "")

            ids.append(chunk_id)
            contents.append(doc.page_content)
            metadatas.append(metadata)

        success = self.vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        return {"add_count": len(ids) if success else 0, "ids": ids, "stored": success}

    @staticmethod
    def _extract_name_candidates(query: str) -> List[str]:
        text = (query or "").strip()
        if not text:
            return []

        cleaned = re.sub(r"[？?。！!，,、：:；;\s]+", "", text)
        if not cleaned:
            return []

        stop_phrases = [
            "请问",
            "请推荐",
            "推荐",
            "介绍一下",
            "帮我介绍",
            "帮我查",
            "查询",
            "是什么",
            "怎么做",
            "如何做",
            "为什么叫",
            "为什么",
            "有什么特色",
            "特色",
            "做法",
            "菜谱",
            "吗",
            "呢",
        ]

        candidates = [cleaned]
        stripped = cleaned
        changed = True
        while changed:
            changed = False
            for phrase in stop_phrases:
                if stripped.startswith(phrase):
                    stripped = stripped[len(phrase):]
                    changed = True
                if stripped.endswith(phrase):
                    stripped = stripped[:-len(phrase)]
                    changed = True

        if stripped and stripped != cleaned:
            candidates.insert(0, stripped)

        unique: List[str] = []
        for candidate in candidates:
            if 2 <= len(candidate) <= 40 and candidate not in unique:
                unique.append(candidate)
        return unique

    @staticmethod
    def _format_recipe_document(recipe: Dict[str, Any]) -> str:
        parts: List[str] = []
        name = recipe.get("name")
        if name:
            parts.append(f"菜名：{name}")

        category = recipe.get("category")
        if category:
            parts.append(f"分类：{category}")

        difficulty = recipe.get("difficulty")
        if difficulty:
            parts.append(f"难度：{difficulty}")

        time_cost = recipe.get("time") or recipe.get("cook_time")
        if time_cost:
            parts.append(f"耗时：{time_cost}")

        ingredients = recipe.get("ingredients") or recipe.get("ingredient_list")
        if ingredients:
            if isinstance(ingredients, list):
                formatted = "、".join(str(item) for item in ingredients)
            else:
                formatted = str(ingredients)
            parts.append(f"食材：{formatted}")

        steps = recipe.get("steps")
        if steps:
            if isinstance(steps, list):
                step_lines = [f"步骤{idx + 1}：{step}" for idx, step in enumerate(steps)]
                parts.extend(step_lines)
            else:
                parts.append(f"步骤：{steps}")

        tips = recipe.get("tips")
        if tips:
            parts.append(f"小贴士：{tips}")

        nutrition = recipe.get("nutrition")
        if nutrition:
            if isinstance(nutrition, dict):
                nutritions = [f"{k}: {v}" for k, v in nutrition.items()]
                parts.append("营养：" + "、".join(nutritions))
            else:
                parts.append(f"营养：{nutrition}")

        return "\n".join(parts)
