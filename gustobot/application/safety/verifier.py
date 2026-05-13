"""Rule-first answer verifier with optional LLM JSON judging."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from gustobot.application.services.llm_client import LLMClient
from gustobot.config import settings

from .schemas import EvidenceItem, PostCheckResult


EVIDENCE_REQUIRED_ROUTES = {"kb-query", "graphrag-query", "text2sql-query", "cypher-query"}


class AnswerVerifier:
    """Validate final answers against collected evidence."""

    def __init__(self, *, use_llm: Optional[bool] = None) -> None:
        self.use_llm = settings.ENABLE_SAFETY_LLM_VERIFIER if use_llm is None else use_llm

    async def verify(
        self,
        query: str,
        answer: str,
        evidence: List[EvidenceItem],
        route: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PostCheckResult:
        rule_result = self._rule_check(query, answer, evidence, route)
        if rule_result.suggested_action in {"retry", "fallback", "refuse"}:
            return rule_result

        if not self.use_llm or not settings.OPENAI_API_KEY:
            return rule_result

        try:
            llm_result = await self._llm_check(query, answer, evidence, route)
            return self._merge_rule_and_llm(rule_result, llm_result)
        except Exception as exc:
            logger.warning("Safety LLM verifier failed; fail-closed to partial/unsupported: {}", exc)
            if evidence:
                return PostCheckResult(
                    verdict="partially_supported",
                    confidence=0.35,
                    issues=["LLM verifier failed; evidence exists but answer was not fully validated"],
                    suggested_action="pass",
                    safe_answer=self._conservative_answer(answer),
                )
            return PostCheckResult(
                verdict="unsupported",
                confidence=0.2,
                issues=["LLM verifier failed and no evidence is available"],
                suggested_action="retry",
            )

    def _rule_check(
        self,
        query: str,
        answer: str,
        evidence: List[EvidenceItem],
        route: Optional[str],
    ) -> PostCheckResult:
        answer = (answer or "").strip()
        evidence_count = len(evidence)

        if not answer:
            return PostCheckResult(
                verdict="invalid_answer",
                confidence=1.0,
                issues=["answer is empty"],
                suggested_action="fallback",
            )

        if self._is_refusal_or_fallback(answer):
            return PostCheckResult(
                verdict="no_evidence" if evidence_count == 0 else "supported",
                confidence=0.85,
                issues=["answer is a conservative fallback/refusal"],
                suggested_action="fallback" if evidence_count == 0 else "pass",
                safe_answer=answer,
            )

        if route in EVIDENCE_REQUIRED_ROUTES and evidence_count == 0:
            return PostCheckResult(
                verdict="no_evidence",
                confidence=0.95,
                issues=[f"route `{route}` requires evidence but none was collected"],
                suggested_action="retry",
            )

        if evidence_count == 0 and self._claims_knowledge_source(answer):
            return PostCheckResult(
                verdict="unsupported",
                confidence=0.95,
                issues=["answer cites knowledge/database/document evidence but no evidence was collected"],
                suggested_action="retry",
            )

        if evidence_count == 0 and route != "general-query" and self._contains_definitive_claim(answer):
            return PostCheckResult(
                verdict="unsupported",
                confidence=0.8,
                issues=["answer contains definitive factual claims without evidence"],
                suggested_action="retry",
            )

        return PostCheckResult(
            verdict="supported",
            confidence=0.65 if evidence_count else 0.45,
            issues=[],
            supported_claims=[answer[:300]] if evidence_count else [],
            suggested_action="pass",
        )

    async def _llm_check(
        self,
        query: str,
        answer: str,
        evidence: List[EvidenceItem],
        route: Optional[str],
    ) -> PostCheckResult:
        evidence_text = self._format_evidence(evidence)
        system_prompt = (
            "You are a strict RAG answer verifier. Judge whether the answer is supported "
            "ONLY by the provided evidence. Do not use outside knowledge. Return strict JSON "
            "with keys: verdict, confidence, issues, supported_claims, unsupported_claims, "
            "suggested_action, safe_answer. Allowed verdicts: supported, partially_supported, "
            "unsupported, unsafe, no_evidence. Allowed suggested_action: pass, retry, fallback, refuse."
        )
        user_message = (
            f"query: {query}\n"
            f"route: {route or ''}\n\n"
            f"answer:\n{answer}\n\n"
            f"evidence:\n{evidence_text or 'NO_EVIDENCE'}\n\n"
            "Return JSON only."
        )
        payload = await LLMClient(temperature=0.0).chat_json(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
        )
        return PostCheckResult.model_validate(payload)

    @staticmethod
    def _merge_rule_and_llm(rule: PostCheckResult, llm: PostCheckResult) -> PostCheckResult:
        if llm.verdict in {"unsupported", "unsafe", "no_evidence", "invalid_answer"}:
            return llm
        if llm.verdict == "partially_supported":
            return llm
        return llm if llm.confidence >= rule.confidence else rule

    @staticmethod
    def _claims_knowledge_source(answer: str) -> bool:
        markers = ["根据知识库", "根据数据库", "根据文档", "根据图谱", "知识库显示", "数据库中"]
        return any(marker in answer for marker in markers)

    @staticmethod
    def _contains_definitive_claim(answer: str) -> bool:
        if len(answer) < 12:
            return False
        definitive_markers = ["是", "为", "共有", "包括", "需要", "步骤", "做法", "传统", "标准"]
        return any(marker in answer for marker in definitive_markers)

    @staticmethod
    def _is_refusal_or_fallback(answer: str) -> bool:
        markers = [
            "没有找到足够可靠的信息",
            "无法给出确定回答",
            "知识库中没有找到",
            "当前资料不足",
            "暂时无法",
            "不能继续执行",
            "请补充",
        ]
        lowered = answer.lower()
        return any(marker in answer for marker in markers) or "no data to summarize" in lowered

    @staticmethod
    def _conservative_answer(answer: str) -> str:
        return f"根据当前资料只能确认以下内容：\n{answer}"

    @staticmethod
    def _format_evidence(evidence: List[EvidenceItem]) -> str:
        chunks: List[str] = []
        for idx, item in enumerate(evidence[:12], 1):
            payload = {
                "source_type": item.source_type,
                "source_name": item.source_name,
                "content": item.content[:800],
            }
            chunks.append(f"[{idx}] {json.dumps(payload, ensure_ascii=False)}")
        return "\n".join(chunks)
