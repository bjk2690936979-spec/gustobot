"""Typed payloads used by the safety defense layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PreDecision = Literal["allow", "deny", "needs_clarification", "risky"]
EvidenceType = Literal[
    "document",
    "kb_chunk",
    "sql_row",
    "cypher_record",
    "tool_output",
    "graph_node",
    "unknown",
]
PostVerdict = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "unsafe",
    "no_evidence",
    "invalid_answer",
]
SuggestedAction = Literal["pass", "retry", "fallback", "refuse"]
FinalStatus = Literal["passed", "partial", "retried_passed", "fallback", "refused"]


class PreCheckResult(BaseModel):
    decision: PreDecision = "allow"
    reason: str = ""
    risk_types: List[str] = Field(default_factory=list)
    suggested_route: Optional[str] = None
    safe_response: Optional[str] = None


class EvidenceItem(BaseModel):
    source_type: EvidenceType = "unknown"
    content: str
    source_name: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostCheckResult(BaseModel):
    verdict: PostVerdict = "supported"
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    suggested_action: SuggestedAction = "pass"
    safe_answer: Optional[str] = None


class SafetyMetricEvent(BaseModel):
    trace_id: Optional[str] = None
    query: str
    route: Optional[str] = None
    pre_check_result: str
    evidence_count: int = 0
    post_check_result: str
    retry_count: int = 0
    final_status: FinalStatus
    hallucination_flag: bool = False
    boundary_violation_flag: bool = False
    refusal_flag: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    extra: Dict[str, Any] = Field(default_factory=dict)
