"""Low-coupling safety helpers for LangGraph internals.

This module adapts the existing SafetyDefenseManager pieces to LangGraph state
without making tool nodes depend on the HTTP chat wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from gustobot.config import settings

from .grounding import GroundingCollector
from .manager import SafetyDefenseManager
from .schemas import EvidenceItem, PostCheckResult
from .verifier import AnswerVerifier


def default_safety_state() -> Dict[str, Any]:
    return {
        "pre_check": None,
        "evidence": [],
        "post_check": None,
        "retry_count": 0,
        "final_status": None,
        "validation_warnings": [],
    }


def normalize_safety_state(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = default_safety_state()
    if isinstance(value, dict):
        for key, item in value.items():
            state[key] = item
    if not isinstance(state.get("evidence"), list):
        state["evidence"] = []
    if not isinstance(state.get("validation_warnings"), list):
        state["validation_warnings"] = []
    return state


def evidence_to_dicts(evidence: Iterable[EvidenceItem | Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[tuple[str, Optional[str], str]] = set()
    for item in evidence:
        if isinstance(item, EvidenceItem):
            payload = item.model_dump()
        elif isinstance(item, dict):
            try:
                payload = EvidenceItem.model_validate(item).model_dump()
            except Exception:
                payload = dict(item)
        else:
            continue
        key = (
            str(payload.get("source_type", "")),
            payload.get("source_name"),
            str(payload.get("content", ""))[:200],
        )
        if key not in seen and str(payload.get("content", "")).strip():
            seen.add(key)
            items.append(payload)
    return items


def evidence_from_payload(payload: Any, route: Optional[str] = None) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        payload = {"tool_outputs": payload}
    return evidence_to_dicts(GroundingCollector().collect(payload, route=route))


def merge_safety_evidence(
    existing_safety: Optional[Dict[str, Any]],
    new_evidence: Iterable[EvidenceItem | Dict[str, Any]],
    *,
    validation_warnings: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    safety = normalize_safety_state(existing_safety)
    merged = evidence_to_dicts([*safety.get("evidence", []), *list(new_evidence)])
    safety["evidence"] = merged
    if validation_warnings:
        current = [str(item) for item in safety.get("validation_warnings", []) if item]
        for warning in validation_warnings:
            warning_text = str(warning)
            if warning_text and warning_text not in current:
                current.append(warning_text)
        safety["validation_warnings"] = current
    return safety


def _post_check_to_dict(post_check: PostCheckResult) -> Dict[str, Any]:
    return post_check.model_dump()


def _status_from_post_check(post_check: PostCheckResult) -> str:
    if post_check.suggested_action == "refuse" or post_check.verdict == "unsafe":
        return "refused"
    if post_check.suggested_action == "fallback" or post_check.verdict in {
        "unsupported",
        "no_evidence",
        "invalid_answer",
    }:
        return "fallback"
    if post_check.verdict == "partially_supported":
        return "partial"
    return "passed"


def create_safety_verify_node(
    verifier: Optional[AnswerVerifier] = None,
    manager: Optional[SafetyDefenseManager] = None,
):
    """Create a LangGraph node that verifies the current generated summary."""

    verifier = verifier or AnswerVerifier()
    manager = manager or SafetyDefenseManager(verifier=verifier)

    async def safety_verify_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        safety = normalize_safety_state(state.get("safety"))

        if not getattr(settings, "ENABLE_SAFETY_DEFENSE", True):
            return {"safety": safety, "steps": ["safety_verify_answer"]}

        query = state.get("question", "")
        answer = state.get("summary") or state.get("answer") or ""
        route = state.get("route_type")
        evidence_payloads = safety.get("evidence", [])
        evidence: List[EvidenceItem] = []
        for item in evidence_payloads:
            try:
                evidence.append(EvidenceItem.model_validate(item))
            except Exception:
                continue

        post_check = await verifier.verify(
            query=query,
            answer=answer,
            evidence=evidence,
            route=route,
            context={"langgraph_internal": True},
        )
        final_status = _status_from_post_check(post_check)

        safe_answer = answer
        if post_check.safe_answer:
            safe_answer = post_check.safe_answer
        elif final_status == "fallback":
            safe_answer = manager.fallback_answer(post_check.verdict)

        safety["post_check"] = _post_check_to_dict(post_check)
        safety["final_status"] = final_status

        update: Dict[str, Any] = {
            "safety": safety,
            "steps": ["safety_verify_answer"],
        }
        if safe_answer != answer:
            update["summary"] = safe_answer
        return update

    return safety_verify_answer
