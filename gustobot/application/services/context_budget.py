"""
Short-term context budgeting for chat sessions.

This module keeps a compact, per-session checkpoint outside the LangGraph state:
rolling summary + the latest N user/assistant turns. Redis is preferred, with an
in-memory fallback so chat flow keeps working when Redis is unavailable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger
from redis.asyncio import Redis

from gustobot.application.services.llm_client import LLMClient
from gustobot.config import settings


ContextMessage = Dict[str, str]
Checkpoint = Dict[str, Any]
SummaryFn = Callable[[str, List[ContextMessage]], Awaitable[str]]

_IN_MEMORY_CHECKPOINTS: Dict[str, Checkpoint] = {}


@dataclass
class ContextBuildResult:
    """Compact context payload prepared for the next LLM/agent call."""

    prompt: str
    summary: str
    recent_messages: List[ContextMessage]
    compressed_context_chars: int
    redis_available: bool


class ContextBudgetManager:
    """Manage rolling summaries and sliding-window context checkpoints."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        window_size: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        summary_trigger_messages: Optional[int] = None,
        summary_max_chars: Optional[int] = None,
        redis_url: Optional[str] = None,
        redis_ttl: Optional[int] = None,
        redis_client: Optional[Redis] = None,
        summarizer: Optional[SummaryFn] = None,
        use_llm_summary: bool = True,
    ) -> None:
        self.enabled = settings.ENABLE_CONTEXT_BUDGET if enabled is None else enabled
        self.window_size = window_size or settings.CONTEXT_WINDOW_SIZE
        self.max_context_chars = max_context_chars or settings.MAX_CONTEXT_CHARS
        self.summary_trigger_messages = (
            summary_trigger_messages or settings.SUMMARY_TRIGGER_MESSAGES
        )
        self.summary_max_chars = summary_max_chars or settings.SUMMARY_MAX_CHARS
        self.redis_ttl = redis_ttl or settings.REDIS_CHECKPOINT_TTL
        self.summarizer = summarizer
        self.use_llm_summary = use_llm_summary
        self._redis_available = False
        self._redis_retry_at = 0.0
        self._redis_retry_interval = 30.0
        self._redis = redis_client or Redis.from_url(
            redis_url or settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )

    async def build_context(self, session_id: str, current_message: str) -> ContextBuildResult:
        """Build the compact context string for the current request."""
        checkpoint = await self._load_checkpoint(session_id)
        summary = str(checkpoint.get("summary") or "")
        recent_messages = self._sanitize_messages(checkpoint.get("recent_messages", []))

        prompt = self._format_context(summary, recent_messages, current_message)
        if self.estimate_context_length(prompt) > self.max_context_chars:
            prompt, recent_messages, summary = self._fit_context_budget(
                summary,
                recent_messages,
                current_message,
            )

        return ContextBuildResult(
            prompt=prompt,
            summary=summary,
            recent_messages=recent_messages,
            compressed_context_chars=self.estimate_context_length(prompt),
            redis_available=self._redis_available,
        )

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> Checkpoint:
        """Append a user/assistant turn and summarize any messages outside the window."""
        checkpoint = await self._load_checkpoint(session_id)
        recent_messages = self._sanitize_messages(checkpoint.get("recent_messages", []))
        recent_messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        checkpoint["recent_messages"] = recent_messages
        checkpoint["session_id"] = session_id
        checkpoint["updated_at"] = self._now()

        checkpoint = await self.summarize_if_needed(session_id, checkpoint=checkpoint)
        await self._save_checkpoint(session_id, checkpoint)
        return checkpoint

    async def summarize_if_needed(
        self,
        session_id: str,
        *,
        checkpoint: Optional[Checkpoint] = None,
    ) -> Checkpoint:
        """Move overflow messages from the sliding window into the rolling summary."""
        checkpoint = checkpoint or await self._load_checkpoint(session_id)
        recent_messages = self._sanitize_messages(checkpoint.get("recent_messages", []))

        max_recent_messages = max(self.window_size, 0) * 2
        if max_recent_messages <= 0 or len(recent_messages) <= max_recent_messages:
            checkpoint["recent_messages"] = recent_messages
            return checkpoint

        if len(recent_messages) <= self.summary_trigger_messages:
            checkpoint["recent_messages"] = recent_messages
            return checkpoint

        overflow_count = len(recent_messages) - max_recent_messages
        overflow_messages = recent_messages[:overflow_count]
        kept_messages = recent_messages[overflow_count:]
        old_summary = str(checkpoint.get("summary") or "")
        new_summary = await self.update_summary(old_summary, overflow_messages)

        checkpoint.update(
            {
                "session_id": session_id,
                "summary": new_summary,
                "recent_messages": kept_messages,
                "updated_at": self._now(),
            }
        )
        return checkpoint

    async def update_summary(
        self,
        old_summary: str,
        overflow_messages: List[ContextMessage],
    ) -> str:
        """Update rolling summary from only the newly overflowed messages."""
        if not overflow_messages:
            return self._truncate(old_summary, self.summary_max_chars)

        if self.summarizer:
            try:
                summary = await self.summarizer(old_summary, overflow_messages)
                return self._truncate(summary.strip(), self.summary_max_chars)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Custom context summarizer failed: {}", exc)

        if self.use_llm_summary and settings.OPENAI_API_KEY:
            try:
                summary = await self._summarize_with_llm(old_summary, overflow_messages)
                return self._truncate(summary.strip(), self.summary_max_chars)
            except Exception as exc:
                logger.warning("LLM context summary failed; using fallback: {}", exc)

        return self._fallback_summary(old_summary, overflow_messages)

    @staticmethod
    def estimate_context_length(text: str) -> int:
        """Estimate context size in characters."""
        return len(text or "")

    async def get_checkpoint(self, session_id: str) -> Checkpoint:
        """Return the current checkpoint for diagnostics/tests."""
        return await self._load_checkpoint(session_id)

    async def clear(self, session_id: str) -> None:
        """Delete a session checkpoint from Redis or fallback memory."""
        key = self._key(session_id)
        if self._should_try_redis():
            try:
                await self._redis.delete(key)
                self._mark_redis_available()
            except Exception:
                self._mark_redis_unavailable()
        _IN_MEMORY_CHECKPOINTS.pop(key, None)

    def is_redis_available(self) -> bool:
        """Return whether the last Redis operation succeeded."""
        return self._redis_available

    async def _summarize_with_llm(
        self,
        old_summary: str,
        overflow_messages: List[ContextMessage],
    ) -> str:
        system_prompt = (
            "你是会话状态摘要器。请把旧摘要和新滑出窗口的对话合并为一个会话状态摘要，"
            "只保留后续有用的信息：用户目标、限制条件、已确认方案、项目名、文件路径、"
            "函数名、报错关键信息、下一步任务。删除寒暄、重复确认和无关闲聊。"
            f"摘要不超过 {self.summary_max_chars} 个中文字符。"
        )
        user_message = (
            f"旧摘要:\n{old_summary or '无'}\n\n"
            f"新滑出窗口的对话:\n{self._format_recent_messages(overflow_messages)}"
        )
        client = LLMClient(temperature=0.2)
        return await client.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.2,
        )

    async def _load_checkpoint(self, session_id: str) -> Checkpoint:
        key = self._key(session_id)
        if self._should_try_redis():
            try:
                raw = await self._redis.get(key)
                self._mark_redis_available()
                if raw:
                    data = json.loads(raw)
                    return self._normalize_checkpoint(session_id, data)
            except Exception as exc:
                if self._redis_available:
                    logger.warning("Redis checkpoint unavailable; using memory fallback: {}", exc)
                self._mark_redis_unavailable()

        return self._normalize_checkpoint(
            session_id,
            _IN_MEMORY_CHECKPOINTS.get(key, {}),
        )

    async def _save_checkpoint(self, session_id: str, checkpoint: Checkpoint) -> None:
        key = self._key(session_id)
        checkpoint = self._normalize_checkpoint(session_id, checkpoint)
        payload = json.dumps(checkpoint, ensure_ascii=False)
        if self._should_try_redis():
            try:
                await self._redis.set(key, payload, ex=self.redis_ttl)
                self._mark_redis_available()
                return
            except Exception as exc:
                if self._redis_available:
                    logger.warning("Failed to save Redis checkpoint; using memory fallback: {}", exc)
                self._mark_redis_unavailable()
        _IN_MEMORY_CHECKPOINTS[key] = checkpoint

    def _normalize_checkpoint(self, session_id: str, data: Checkpoint) -> Checkpoint:
        return {
            "session_id": str(data.get("session_id") or session_id),
            "summary": self._truncate(str(data.get("summary") or ""), self.summary_max_chars),
            "recent_messages": self._sanitize_messages(data.get("recent_messages", [])),
            "updated_at": str(data.get("updated_at") or self._now()),
        }

    @staticmethod
    def _sanitize_messages(messages: Any) -> List[ContextMessage]:
        sanitized: List[ContextMessage] = []
        if not isinstance(messages, list):
            return sanitized
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "")
            if content:
                sanitized.append({"role": role, "content": content})
        return sanitized

    def _fit_context_budget(
        self,
        summary: str,
        recent_messages: List[ContextMessage],
        current_message: str,
    ) -> tuple[str, List[ContextMessage], str]:
        trimmed_recent = list(recent_messages)
        trimmed_summary = self._truncate(summary, self.summary_max_chars)
        prompt = self._format_context(trimmed_summary, trimmed_recent, current_message)

        while (
            trimmed_recent
            and self.estimate_context_length(prompt) > self.max_context_chars
        ):
            trimmed_recent = trimmed_recent[1:]
            prompt = self._format_context(trimmed_summary, trimmed_recent, current_message)

        if self.estimate_context_length(prompt) <= self.max_context_chars:
            return prompt, trimmed_recent, trimmed_summary

        summary_budget = max(0, self.max_context_chars // 8)
        trimmed_summary = self._truncate(trimmed_summary, summary_budget)
        prompt = self._format_context(trimmed_summary, trimmed_recent, current_message)
        if self.estimate_context_length(prompt) <= self.max_context_chars:
            return prompt, trimmed_recent, trimmed_summary

        fixed_overhead = self.estimate_context_length(
            self._format_context(trimmed_summary, trimmed_recent, "")
        )
        current_budget = max(0, self.max_context_chars - fixed_overhead)
        trimmed_current = self._truncate(current_message, current_budget)
        prompt = self._format_context(trimmed_summary, trimmed_recent, trimmed_current)
        return prompt[: self.max_context_chars], trimmed_recent, trimmed_summary

    def _fallback_summary(
        self,
        old_summary: str,
        overflow_messages: List[ContextMessage],
    ) -> str:
        useful_parts: List[str] = []
        if old_summary.strip():
            useful_parts.append(old_summary.strip())

        for message in overflow_messages:
            role = "用户" if message["role"] == "user" else "助手"
            content = self._single_line(message["content"])
            if not content:
                continue
            useful_parts.append(f"{role}: {content}")

        combined = "\n".join(useful_parts)
        return self._truncate(combined, self.summary_max_chars)

    def _format_context(
        self,
        summary: str,
        recent_messages: List[ContextMessage],
        current_message: str,
    ) -> str:
        return (
            "[历史摘要]\n"
            f"{summary.strip() or '无'}\n\n"
            "[最近对话]\n"
            f"{self._format_recent_messages(recent_messages) or '无'}\n\n"
            "[当前问题]\n"
            f"{current_message}"
        )

    @staticmethod
    def _format_recent_messages(messages: List[ContextMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            role = "user" if message["role"] == "user" else "assistant"
            content = ContextBudgetManager._single_line(message["content"])
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _single_line(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        text = text or ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 20:
            return text[:max_chars]
        head = max_chars // 2
        tail = max_chars - head - 3
        return f"{text[:head]}...{text[-tail:]}"

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:checkpoint:{session_id}"

    def _should_try_redis(self) -> bool:
        return time.monotonic() >= self._redis_retry_at

    def _mark_redis_available(self) -> None:
        self._redis_available = True
        self._redis_retry_at = 0.0

    def _mark_redis_unavailable(self) -> None:
        self._redis_available = False
        self._redis_retry_at = time.monotonic() + self._redis_retry_interval


_default_manager: Optional[ContextBudgetManager] = None


def get_context_budget_manager() -> ContextBudgetManager:
    """Return the process-wide context budget manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ContextBudgetManager()
    return _default_manager
