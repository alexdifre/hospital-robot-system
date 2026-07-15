"""Unified Planning engine selection for PDDL task planning."""

from __future__ import annotations

from typing import Optional


DEFAULT_PDDL_PLANNING_ENGINE = "enhsp-opt"
OPTIMAL_PLAN_STATUS = "SOLVED_OPTIMALLY"


def normalize_pddl_planning_engine(engine_name: Optional[str]) -> str:
    """Return the required optimal engine and reject satisficing variants."""
    normalized = (
        DEFAULT_PDDL_PLANNING_ENGINE
        if engine_name is None
        else engine_name.strip().lower()
    )
    if normalized != DEFAULT_PDDL_PLANNING_ENGINE:
        raise ValueError(
            "This system requires the optimal PDDL planner 'enhsp-opt'; "
            f"received {engine_name!r}."
        )
    return normalized


def is_optimal_plan_status(status) -> bool:
    """Return True only when Unified Planning certifies an optimal plan."""
    status_name = getattr(status, "name", None)
    if status_name is None:
        status_name = str(status).rsplit(".", 1)[-1]
    return status_name == OPTIMAL_PLAN_STATUS


def make_pddl_oneshot_planner(engine_name: Optional[str] = None):
    """Create ENHSP-opt, failing clearly when the optimal engine is unavailable."""
    from unified_planning.shortcuts import OneshotPlanner

    engine = normalize_pddl_planning_engine(engine_name)
    try:
        return OneshotPlanner(name=engine)
    except Exception as exc:
        if exc.__class__.__name__ in {
            "UPNoRequestedEngineAvailableException",
            "UPNoSuitableEngineAvailableException",
        }:
            raise RuntimeError(
                "The required optimal planner 'enhsp-opt' is unavailable. "
                "Install unified-planning[enhsp] and a Java runtime."
            ) from exc
        raise
