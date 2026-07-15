import sys
import types

import pytest

from core.task_planning.pddl_engine import (
    DEFAULT_PDDL_PLANNING_ENGINE,
    is_optimal_plan_status,
    make_pddl_oneshot_planner,
    normalize_pddl_planning_engine,
)


def test_pddl_planning_engine_defaults_to_enhsp_opt():
    assert DEFAULT_PDDL_PLANNING_ENGINE == "enhsp-opt"
    assert normalize_pddl_planning_engine(None) == "enhsp-opt"


def test_pddl_planning_engine_accepts_only_enhsp_opt():
    assert normalize_pddl_planning_engine("enhsp-opt") == "enhsp-opt"
    with pytest.raises(ValueError, match="requires the optimal PDDL planner"):
        normalize_pddl_planning_engine("enhsp")


def test_only_solved_optimally_status_is_accepted():
    assert is_optimal_plan_status("PlanGenerationResultStatus.SOLVED_OPTIMALLY")
    assert not is_optimal_plan_status("PlanGenerationResultStatus.SOLVED_SATISFICING")
    assert not is_optimal_plan_status("SOLVED")


def test_make_pddl_oneshot_planner_requests_enhsp_opt(monkeypatch):
    captured = {}

    class FakeOneshotPlanner:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    fake_shortcuts = types.ModuleType("unified_planning.shortcuts")
    fake_shortcuts.OneshotPlanner = FakeOneshotPlanner
    monkeypatch.setitem(sys.modules, "unified_planning.shortcuts", fake_shortcuts)

    planner = make_pddl_oneshot_planner("enhsp-opt")

    assert isinstance(planner, FakeOneshotPlanner)
    assert captured["kwargs"]["name"] == "enhsp-opt"
