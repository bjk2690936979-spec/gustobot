from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

try:
    import pytest
except ModuleNotFoundError:  # Allow `python tests/test_context_budget.py` without pytest.
    class _Mark:
        @staticmethod
        def asyncio(func):
            return func

    class _PytestStub:
        mark = _Mark()

    pytest = _PytestStub()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DEBUG"] = "true"

from gustobot.application.services.context_budget import ContextBudgetManager


def _long_user_message(turn: int) -> str:
    return (
        f"第 {turn} 轮用户需求：继续优化 GustoBot 上下文工程，项目名 GustoBot，"
        "限制条件是不改 Router 和 RAG 主流程，只保留短期会话状态。"
        "请关注 session_id 隔离、Redis checkpoint、滑动窗口和 token 预算。"
    )


def _long_assistant_message(turn: int) -> str:
    return (
        f"第 {turn} 轮助手回复：已确认旁路方案，相关文件包括 "
        "gustobot/interfaces/http/v1/chat.py 与 gustobot/application/services/context_budget.py。"
        "下一步是保存 user-assistant turn 并只总结滑出窗口的旧消息。"
    )


def _format_original_context(turns: int, current_message: str) -> str:
    lines = []
    for idx in range(1, turns + 1):
        lines.append(f"user: {_long_user_message(idx)}")
        lines.append(f"assistant: {_long_assistant_message(idx)}")
    lines.append(f"current: {current_message}")
    return "\n".join(lines)


async def run_context_budget_demo() -> Dict[str, Any]:
    manager = ContextBudgetManager(
        enabled=True,
        window_size=6,
        max_context_chars=12000,
        summary_trigger_messages=12,
        summary_max_chars=500,
        redis_url="redis://127.0.0.1:1/0",
        use_llm_summary=False,
    )
    session_a = f"context-budget-a-{uuid.uuid4()}"
    session_b = f"context-budget-b-{uuid.uuid4()}"
    current_message = "当前问题：请继续基于前面的方案完成最小可用实现。"

    await manager.clear(session_a)
    await manager.clear(session_b)

    for idx in range(1, 13):
        await manager.save_turn(
            session_a,
            _long_user_message(idx),
            _long_assistant_message(idx),
        )

    checkpoint_a = await manager.get_checkpoint(session_a)
    context_a = await manager.build_context(session_a, current_message)

    await manager.save_turn(
        session_b,
        "session_b 用户只关心糖醋排骨做法，不应混入 session_a 的工程上下文。",
        "session_b 助手只回答糖醋排骨，不包含 GustoBot 工程文件路径。",
    )
    checkpoint_b = await manager.get_checkpoint(session_b)

    session_a_blob = json.dumps(checkpoint_a, ensure_ascii=False)
    session_b_blob = json.dumps(checkpoint_b, ensure_ascii=False)
    session_isolation = (
        "糖醋排骨" not in session_a_blob
        and "context_budget.py" not in session_b_blob
    )

    fallback_manager = ContextBudgetManager(
        enabled=True,
        redis_url="redis://127.0.0.1:1/0",
        use_llm_summary=False,
    )
    fallback_session = f"context-budget-fallback-{uuid.uuid4()}"
    await fallback_manager.save_turn(
        fallback_session,
        "Redis 不可用时也要保存这一轮用户输入。",
        "已降级到内存 checkpoint，主流程继续。",
    )
    fallback_context = await fallback_manager.build_context(
        fallback_session,
        "验证 fallback 是否正常。",
    )

    original_context_chars = len(_format_original_context(12, current_message))
    compressed_context_chars = context_a.compressed_context_chars
    reduction_percent = round(
        (1 - compressed_context_chars / original_context_chars) * 100,
        2,
    )

    metrics = {
        "original_context_chars": original_context_chars,
        "compressed_context_chars": compressed_context_chars,
        "reduction_percent": reduction_percent,
        "recent_message_count": len(checkpoint_a["recent_messages"]),
        "summary_generated": bool(checkpoint_a["summary"]),
        "session_isolation_result": session_isolation,
        "redis_available": fallback_context.redis_available,
    }

    assert len(checkpoint_a["recent_messages"]) == 12
    assert checkpoint_a["summary"]
    assert context_a.compressed_context_chars < manager.max_context_chars
    assert session_isolation is True
    assert fallback_context.redis_available is False
    assert "Redis 不可用" in json.dumps(
        await fallback_manager.get_checkpoint(fallback_session),
        ensure_ascii=False,
    )

    return metrics


@pytest.mark.asyncio
async def test_context_budget_manager() -> None:
    metrics = await run_context_budget_demo()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_context_budget_demo()), ensure_ascii=False, indent=2))
