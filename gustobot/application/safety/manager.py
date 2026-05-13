"""Lightweight safety defense manager for the main chat path."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gustobot.config import settings

from .boundary_checker import BoundaryChecker
from .grounding import GroundingCollector
from .metrics import SafetyMetricsWriter
from .schemas import EvidenceItem, PostCheckResult, PreCheckResult, SafetyMetricEvent
from .verifier import AnswerVerifier


class SafetyDefenseManager:
    """Pre-check, evidence collection, post-check, retry policy, and metrics."""

    def __init__(
        self,
        *,
        boundary_checker: Optional[BoundaryChecker] = None,
        grounding_collector: Optional[GroundingCollector] = None,
        verifier: Optional[AnswerVerifier] = None,
        metrics_writer: Optional[SafetyMetricsWriter] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.boundary_checker = boundary_checker or BoundaryChecker()
        self.grounding_collector = grounding_collector or GroundingCollector()
        self.verifier = verifier or AnswerVerifier()
        self.metrics_writer = metrics_writer or SafetyMetricsWriter()
        self.max_retries = settings.SAFETY_MAX_RETRIES if max_retries is None else max_retries

    async def pre_check(self, query: str, context: Optional[Dict[str, Any]] = None) -> PreCheckResult:
        return await self.boundary_checker.check(query, context)

    def collect_evidence(self, result: Dict[str, Any], route: Optional[str] = None) -> List[EvidenceItem]:
        return self.grounding_collector.collect(result or {}, route=route)

    async def post_check(
        self,
        query: str,
        answer: str,
        evidence: List[EvidenceItem],
        route: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PostCheckResult:
        return await self.verifier.verify(query, answer, evidence, route, context)

    def should_retry(self, post_check_result: PostCheckResult, retry_count: int) -> bool:
        if retry_count >= self.max_retries:
            return False
        retryable_verdicts = {"unsupported", "no_evidence", "partially_supported", "unsafe"}
        return (
            post_check_result.verdict in retryable_verdicts
            and post_check_result.suggested_action == "retry"
        )

    def build_retry_query(self, original_query: str, post_check_result: PostCheckResult) -> str:
        if post_check_result.verdict in {"unsupported", "no_evidence"}:
            return (
                f"请重新检索并回答：{original_query}\n"
                "只输出有知识库、数据库或工具结果支撑的内容；没有证据就回答无法确认。"
            )
        return (
            f"{original_query}\n"
            "请只基于检索到的知识库、数据库或工具结果回答。"
            "若没有可靠证据，请明确说明当前资料不足，不要编造。"
        )

    @staticmethod
    def fallback_answer(reason: Optional[str] = None) -> str:
        suffix = f"（原因：{reason}）" if reason else ""
        return f"当前知识库中没有找到足够可靠的信息，无法给出确定回答。{suffix}"

    def record_metrics(self, event: SafetyMetricEvent) -> None:
        self.metrics_writer.write(event)

    @staticmethod
    def metric_event(
        *,
        trace_id: Optional[str],
        query: str,
        route: Optional[str],
        pre_check: PreCheckResult,
        evidence_count: int,
        post_check: Optional[PostCheckResult],
        retry_count: int,
        final_status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SafetyMetricEvent:
        post_result = post_check.verdict if post_check else "not_run"
        hallucination_flag = (
            post_result in {"unsupported", "no_evidence"}
            and final_status not in {"fallback", "refused"}
        )
        boundary_violation_flag = pre_check.decision in {"deny", "risky"} or bool(pre_check.risk_types)
        refusal_flag = final_status == "refused"
        return SafetyMetricEvent(
            trace_id=trace_id,
            query=query,
            route=route,
            pre_check_result=pre_check.decision,
            evidence_count=evidence_count,
            post_check_result=post_result,
            retry_count=retry_count,
            final_status=final_status,  # type: ignore[arg-type]
            hallucination_flag=hallucination_flag,
            boundary_violation_flag=boundary_violation_flag,
            refusal_flag=refusal_flag,
            extra=extra or {},
        )


_default_manager: Optional[SafetyDefenseManager] = None


def get_safety_defense_manager() -> SafetyDefenseManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SafetyDefenseManager()
    return _default_manager
