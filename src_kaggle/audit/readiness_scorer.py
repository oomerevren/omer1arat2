"""Component readiness scoring."""
from __future__ import annotations
from typing import Any


def status_from_score(score: int) -> str:
    if score >= 5:
        return "green"
    if score >= 3:
        return "yellow"
    return "red"


def component_row(name: str, score: int, technical_risk: str, private_lb_impact: str, submission_day_impact: str, confidence: str, evidence: list[str], gaps: list[str], action: str, priority: str) -> dict[str, Any]:
    return {
        "component_name": name,
        "status": status_from_score(score),
        "readiness_score": score,
        "technical_risk": technical_risk,
        "private_lb_impact": private_lb_impact,
        "submission_day_impact": submission_day_impact,
        "confidence_level": confidence,
        "evidence": " | ".join(evidence),
        "open_gaps": " | ".join(gaps) if gaps else "none",
        "recommended_action": action,
        "priority": priority,
    }
