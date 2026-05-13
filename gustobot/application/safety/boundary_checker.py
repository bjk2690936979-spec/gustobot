"""Rule-first query boundary checks for the chat entrypoint."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .schemas import PreCheckResult

GuardrailFn = Callable[[str, Optional[Dict[str, Any]]], Awaitable[PreCheckResult]]


class BoundaryChecker:
    """Detect high-risk, vague, or out-of-scope queries before tools run."""

    DENY_RESPONSE = "这个请求涉及越权、破坏性操作或敏感信息泄露，我不能继续执行。"
    CLARIFY_RESPONSE = "你的问题还不够明确，请补充你想查询的对象或范围。"

    _danger_patterns = [
        r"drop\s+table",
        r"truncate\s+table",
        r"delete\s+from",
        r"rm\s+-rf",
        r"删除所有",
        r"清空.*数据库",
        r"泄露.*(密钥|密码|token|api[_ -]?key)",
        r"数据库密码",
        r"恶意代码",
        r"越权访问",
    ]

    _injection_patterns = [
        r"忽略.*(之前|以上|所有).*(规则|指令|限制)",
        r"输出.*系统提示词",
        r"显示.*system prompt",
        r"绕过.*(安全|限制|规则)",
        r"ignore (all )?(previous|prior) instructions",
        r"reveal.*(system prompt|developer message)",
    ]

    _business_keywords = [
        "菜",
        "菜谱",
        "食材",
        "烹饪",
        "做法",
        "步骤",
        "口味",
        "鲁菜",
        "川菜",
        "粤菜",
        "火锅",
        "豆腐",
        "番茄",
        "知识库",
        "数据库",
        "图谱",
        "统计",
        "SQL",
        "Cypher",
        "GustoBot",
        "项目",
        "文件",
        "代码",
    ]

    _general_keywords = ["你好", "您好", "谢谢", "介绍一下你", "你是谁", "help"]

    _vague_queries = {"说一下", "介绍一下", "这个", "那个", "随便", "帮我看看"}

    def __init__(self, llm_guardrail: Optional[GuardrailFn] = None, *, use_llm: bool = False) -> None:
        self.llm_guardrail = llm_guardrail
        self.use_llm = use_llm

    async def check(self, query: str, context: Optional[Dict[str, Any]] = None) -> PreCheckResult:
        query = (query or "").strip()
        if not query:
            return PreCheckResult(
                decision="needs_clarification",
                reason="empty query",
                safe_response=self.CLARIFY_RESPONSE,
            )

        lowered = query.lower()
        risks: List[str] = []

        if self._matches_any(query, lowered, self._danger_patterns):
            return PreCheckResult(
                decision="deny",
                reason="destructive or sensitive request",
                risk_types=["destructive_operation"],
                suggested_route="blocked",
                safe_response=self.DENY_RESPONSE,
            )

        if self._matches_any(query, lowered, self._injection_patterns):
            return PreCheckResult(
                decision="risky",
                reason="prompt injection or policy bypass attempt",
                risk_types=["prompt_injection"],
                suggested_route="blocked",
                safe_response=self.DENY_RESPONSE,
            )

        if self._looks_vague(query):
            return PreCheckResult(
                decision="needs_clarification",
                reason="query is too vague",
                risk_types=["ambiguous_query"],
                safe_response=self.CLARIFY_RESPONSE,
            )

        if not self._is_business_related(query) and not self._is_general_chat(query):
            return PreCheckResult(
                decision="allow",
                reason="outside domain; route to general chat only",
                risk_types=["business_unrelated"],
                suggested_route="general-query",
            )

        if self.use_llm and self.llm_guardrail:
            try:
                return await self.llm_guardrail(query, context)
            except Exception:
                return PreCheckResult(
                    decision="risky",
                    reason="LLM boundary check failed; fail-closed as risky",
                    risk_types=["guardrail_llm_failed"],
                    suggested_route="blocked",
                    safe_response=self.DENY_RESPONSE,
                )

        return PreCheckResult(decision="allow", reason="rule check passed", risk_types=risks)

    @staticmethod
    def _matches_any(query: str, lowered: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, lowered, re.IGNORECASE) or re.search(pattern, query) for pattern in patterns)

    def _is_business_related(self, query: str) -> bool:
        return any(keyword.lower() in query.lower() for keyword in self._business_keywords)

    def _is_general_chat(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword.lower() in lowered for keyword in self._general_keywords)

    def _looks_vague(self, query: str) -> bool:
        normalized = query.strip()
        if len(normalized) <= 1:
            return True
        return normalized in self._vague_queries
