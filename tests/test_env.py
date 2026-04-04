"""
test_env.py — Supply Chain Sim Environment Test Suite
=======================================================
Covers:
  Phase 1 — Python import sanity
  Phase 2 — Offline unit logic (tasks, grader, models, action schema)
  Phase 3 — Local server smoke test (requires server running on localhost:8000)

Run:
    # Phases 1 & 2 only (no server needed):
    pytest tests/test_env.py -v -m "not server"

    # All phases (server must be running):
    uvicorn server.app:app --port 8000 &
    pytest tests/test_env.py -v
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

# ── Shared fixtures ────────────────────────────────────────────────────────

BASE_URL = "http://localhost:7860"

GOOD_OBS = {
    "inventory_levels": {
        "electronics": 800,
        "apparel": 700,
        "food": 900,
        "medical": 850,
        "automotive": 750,
    },
    "supplier_reliability": {
        "supplier_A": 0.95,
        "supplier_B": 0.92,
        "supplier_C": 0.88,
        "supplier_D": 0.91,
    },
    "service_level": 0.85,
    "cash_balance": 50_000.0,
    "period": 5,
    "disruption_active": False,
    "disruption_type": None,
    "current_costs": 10_000.0,
    "message": "",
    "pending_orders": [],
    "demand_forecast": {},
}


def _http_get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _http_post(path: str, data: Any) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    raw = json.dumps(data).encode()
    req.add_header("Content-Length", str(len(raw)))
    with urllib.request.urlopen(req, data=raw, timeout=10) as r:
        return json.loads(r.read())


def _server_is_up() -> bool:
    try:
        _http_get("/health")
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Import Sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase1Imports:
    """Phase 1: All critical packages and modules must import cleanly."""

    def test_fastapi_importable(self):
        import fastapi  # noqa: F401
        assert fastapi.__version__

    def test_uvicorn_importable(self):
        import uvicorn  # noqa: F401

    def test_pydantic_importable(self):
        import pydantic
        assert int(pydantic.__version__.split(".")[0]) >= 2, "Pydantic v2+ required"

    def test_numpy_importable(self):
        import numpy as np  # noqa: F401
        assert np.__version__

    def test_scipy_importable(self):
        import scipy  # noqa: F401

    def test_openai_importable(self):
        from openai import OpenAI  # noqa: F401

    def test_openenv_core_importable(self):
        """Critical: the most common failure point."""
        from openenv.core.env_server import create_fastapi_app  # noqa: F401

    def test_openenv_core_types_importable(self):
        from openenv.core.env_server.types import Action, Observation  # noqa: F401

    def test_server_models_importable(self):
        from server.models import (  # noqa: F401
            SupplyChainAction,
            SupplyChainObservation,
            TaskResult,
        )

    def test_server_tasks_importable(self):
        from server.tasks import TASK_REGISTRY, get_task  # noqa: F401

    def test_server_grader_importable(self):
        from server.grader import grade_task, run_all_graders  # noqa: F401

    def test_server_environment_importable(self):
        from server.environment import SupplyChainEnvironment  # noqa: F401

    def test_server_dynamics_importable(self):
        import server.dynamics  # noqa: F401

    def test_server_events_importable(self):
        import server.events  # noqa: F401

    def test_server_reward_importable(self):
        import server.reward  # noqa: F401

    def test_server_app_importable(self):
        from server.app import app  # noqa: F401
        assert app is not None


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Offline Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase2TaskRegistry:
    """Phase 2a: Task registry completeness and shape."""

    def setup_method(self):
        from server.tasks import TASK_REGISTRY
        self.registry = TASK_REGISTRY

    def test_all_three_tasks_present(self):
        expected = {"inventory_management", "supplier_negotiation", "disruption_response"}
        assert set(self.registry.keys()) == expected

    def test_each_task_has_required_fields(self):
        required = ["task_id", "difficulty", "goal", "max_steps", "initial_cash", "success_criteria"]
        for tid, cfg in self.registry.items():
            for field in required:
                assert hasattr(cfg, field), f"Task '{tid}' missing field: {field}"

    def test_difficulties_are_valid(self):
        valid = {"easy", "medium", "hard"}
        for tid, cfg in self.registry.items():
            assert cfg.difficulty in valid, f"'{tid}' has invalid difficulty: {cfg.difficulty}"

    def test_max_steps_positive(self):
        for tid, cfg in self.registry.items():
            assert cfg.max_steps > 0, f"'{tid}' max_steps must be positive"

    def test_initial_cash_positive(self):
        for tid, cfg in self.registry.items():
            assert cfg.initial_cash > 0, f"'{tid}' initial_cash must be positive"

    def test_inventory_management_config(self):
        cfg = self.registry["inventory_management"]
        assert cfg.success_criteria["service_level_threshold"] == 0.80
        assert cfg.success_criteria["consecutive_periods"] == 10
        assert cfg.disruptions_enabled is False

    def test_supplier_negotiation_config(self):
        cfg = self.registry["supplier_negotiation"]
        assert cfg.success_criteria["reliability_target"] == 0.92
        assert cfg.success_criteria["min_cash_balance"] == 200_000.0

    def test_disruption_response_config(self):
        cfg = self.registry["disruption_response"]
        assert cfg.success_criteria["service_level_threshold"] == 0.75
        assert cfg.success_criteria["recovery_window"] == 5
        assert 3 in cfg.success_criteria["disruption_periods"]
        assert 8 in cfg.success_criteria["disruption_periods"]
        assert cfg.forced_events[3] == "port_strike"
        assert cfg.forced_events[8] == "supplier_failure"

    def test_get_task_helper(self):
        from server.tasks import get_task
        cfg = get_task("inventory_management")
        assert cfg.task_id == "inventory_management"

    def test_get_task_unknown_raises(self):
        from server.tasks import get_task
        with pytest.raises(KeyError):
            get_task("nonexistent_task")


class TestPhase2ActionSchema:
    """Phase 2b: SupplyChainAction model validation."""

    def setup_method(self):
        from server.models import SupplyChainAction
        self.Action = SupplyChainAction

    def test_hold_action(self):
        a = self.Action(action_type="hold")
        assert a.action_type == "hold"
        assert a.product_id is None
        assert a.supplier_id is None

    def test_order_action(self):
        a = self.Action(
            action_type="order",
            product_id="electronics",
            supplier_id="supplier_A",
            quantity=500,
        )
        assert a.action_type == "order"
        assert a.product_id == "electronics"
        assert a.quantity == 500

    def test_negotiate_action(self):
        a = self.Action(action_type="negotiate", supplier_id="supplier_B")
        assert a.action_type == "negotiate"
        assert a.supplier_id == "supplier_B"

    def test_emergency_source_action(self):
        a = self.Action(action_type="emergency_source", product_id="food", quantity=300)
        assert a.action_type == "emergency_source"
        assert a.quantity == 300

    def test_reroute_action(self):
        a = self.Action(action_type="reroute")
        assert a.action_type == "reroute"

    def test_action_type_is_required(self):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, TypeError)):
            self.Action()  # missing action_type

    def test_notes_field_optional(self):
        a = self.Action(action_type="hold", notes="reasoning text")
        assert a.notes == "reasoning text"


class TestPhase2GraderInventory:
    """Phase 2c: Grader — inventory_management."""

    def setup_method(self):
        from server.grader import grade_task
        self.grade = grade_task

    def _make_obs(self, service_level: float, period: int = 0) -> dict:
        obs = dict(GOOD_OBS)
        obs["service_level"] = service_level
        obs["period"] = period
        return obs

    def test_empty_trajectory_returns_zero(self):
        result = self.grade("inventory_management", [])
        assert result.score == 0.0
        assert result.passed is False

    def test_all_above_threshold_scores_high(self):
        traj = [self._make_obs(0.90, i) for i in range(10)]
        result = self.grade("inventory_management", traj)
        assert result.score > 0.5
        assert 0.0 <= result.score <= 1.0

    def test_all_below_threshold_scores_zero(self):
        traj = [self._make_obs(0.50, i) for i in range(10)]
        result = self.grade("inventory_management", traj)
        assert result.score == 0.0
        assert result.passed is False

    def test_score_is_clamped(self):
        traj = [self._make_obs(1.0, i) for i in range(15)]
        result = self.grade("inventory_management", traj)
        assert 0.0 <= result.score <= 1.0

    def test_result_has_task_id(self):
        result = self.grade("inventory_management", [self._make_obs(0.85)])
        assert result.task_id == "inventory_management"

    def test_result_has_details(self):
        traj = [self._make_obs(0.85, i) for i in range(5)]
        result = self.grade("inventory_management", traj)
        assert "periods_above_threshold" in result.details  # backward compat
        assert "longest_streak" in result.details           # consecutive tracking
        assert "threshold" in result.details


class TestPhase2GraderSupplierNegotiation:
    """Phase 2d: Grader — supplier_negotiation."""

    def setup_method(self):
        from server.grader import grade_task
        self.grade = grade_task

    def _make_obs(self, mean_rel: float, cash: float) -> dict:
        obs = dict(GOOD_OBS)
        # Set all suppliers to the same mean_rel
        obs["supplier_reliability"] = {
            "supplier_A": mean_rel,
            "supplier_B": mean_rel,
            "supplier_C": mean_rel,
            "supplier_D": mean_rel,
        }
        obs["cash_balance"] = cash
        return obs

    def test_empty_trajectory_returns_zero(self):
        result = self.grade("supplier_negotiation", [])
        assert result.score == 0.0

    def test_hold_agent_scores_partial_cash(self):
        """Agent that never negotiates: reliability stays at or below initial mean (0.8625).
        rel_score = 0.0. Cash >= min_cash → cash_score = 1.0. Total = 0.30 * 1.0 = 0.30."""
        # Use 0.86 < 0.8625 (initial mean) to ensure rel_score = 0
        traj = [self._make_obs(0.86, 250_000)] * 5
        result = self.grade("supplier_negotiation", traj)
        assert result.score == pytest.approx(0.30, abs=1e-9), f"Expected 0.30, got {result.score}"

    def test_perfect_reliability_and_cash_scores_high(self):
        traj = [self._make_obs(0.95, 250_000)] * 5
        result = self.grade("supplier_negotiation", traj)
        assert result.score > 0.5

    def test_bankrupt_agent_gets_no_cash_score(self):
        """Even with good reliability, negative cash drags score."""
        traj = [self._make_obs(0.95, -1)] * 5
        result = self.grade("supplier_negotiation", traj)
        # Cash score = 0, so total is 70% of rel_score only
        assert result.score < 1.0

    def test_score_is_clamped(self):
        traj = [self._make_obs(0.99, 500_000)] * 10
        result = self.grade("supplier_negotiation", traj)
        assert 0.0 <= result.score <= 1.0


class TestPhase2GraderDisruptionResponse:
    """Phase 2e: Grader — disruption_response."""

    def setup_method(self):
        from server.grader import grade_task
        self.grade = grade_task

    def _make_obs(self, period: int, service_level: float) -> dict:
        obs = dict(GOOD_OBS)
        obs["period"] = period
        obs["service_level"] = service_level
        return obs

    def test_empty_trajectory_returns_zero(self):
        result = self.grade("disruption_response", [])
        assert result.score == 0.0

    def test_no_recovery_window_obs_scores_zero(self):
        """Trajectory has no periods in the windows after disruptions."""
        traj = [self._make_obs(0, 0.85), self._make_obs(1, 0.85), self._make_obs(2, 0.85)]
        result = self.grade("disruption_response", traj)
        assert result.score == 0.0

    def test_full_recovery_after_both_disruptions(self):
        """Simulate full recovery in windows [4-8] and [9-13]."""
        traj = []
        # Pre-disruption
        for p in range(4):
            traj.append(self._make_obs(p, 0.85))
        # Recovery window after d_period=3: periods 4,5,6,7,8
        for p in range(4, 9):
            traj.append(self._make_obs(p, 0.85))
        # Intermediate
        traj.append(self._make_obs(9, 0.85))
        # Recovery window after d_period=8: periods 9,10,11,12,13
        for p in range(9, 14):
            traj.append(self._make_obs(p, 0.85))

        result = self.grade("disruption_response", traj)
        assert result.score > 0.0
        assert 0.0 <= result.score <= 1.0

    def test_result_has_recovery_details(self):
        traj = [self._make_obs(i, 0.80) for i in range(15)]
        result = self.grade("disruption_response", traj)
        assert "recovery_scores" in result.details
        assert "disruption_periods" in result.details


class TestPhase2RunAllGraders:
    """Phase 2f: run_all_graders bulk function."""

    def test_run_all_graders_returns_all_tasks(self):
        from server.grader import run_all_graders
        trajectories = {
            "inventory_management": [GOOD_OBS] * 5,
            "supplier_negotiation": [GOOD_OBS] * 5,
            "disruption_response": [GOOD_OBS] * 5,
        }
        results = run_all_graders(trajectories)
        assert set(results.keys()) == set(trajectories.keys())

    def test_run_all_graders_scores_are_valid(self):
        from server.grader import run_all_graders
        trajectories = {
            "inventory_management": [GOOD_OBS] * 5,
            "supplier_negotiation": [GOOD_OBS] * 5,
            "disruption_response": [GOOD_OBS] * 5,
        }
        results = run_all_graders(trajectories)
        for tid, result in results.items():
            assert 0.0 <= result.score <= 1.0, f"{tid}: score out of range"

    def test_unknown_task_returns_zero(self):
        from server.grader import grade_task
        result = grade_task("nonexistent_task_xyz", [GOOD_OBS])
        assert result.score == 0.0
        assert result.passed is False

    def test_grader_is_deterministic(self):
        """Same input must always produce same output (seed=42 guarantee)."""
        from server.grader import grade_task
        traj = [GOOD_OBS] * 8
        r1 = grade_task("inventory_management", traj)
        r2 = grade_task("inventory_management", traj)
        assert r1.score == r2.score


class TestPhase2ModelObservation:
    """Phase 2g: SupplyChainObservation model."""

    def setup_method(self):
        from server.models import SupplyChainObservation
        self.Obs = SupplyChainObservation

    def test_defaults_work(self):
        obs = self.Obs()
        assert obs.service_level == 1.0
        assert obs.disruption_active is False
        assert obs.disruption_type is None
        assert obs.period == 0

    def test_full_construction(self):
        obs = self.Obs(**GOOD_OBS)
        assert obs.service_level == 0.85
        assert obs.cash_balance == 50_000.0


class TestPhase2InferenceHelpers:
    """Phase 2h: Helpers in inference.py."""

    def setup_method(self):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "inference",
            "inference.py",
        )
        self.mod = importlib.util.load_from_spec = spec
        # Import directly
        sys.path.insert(0, ".")
        import inference
        self.inf = inference

    def test_extract_action_valid_json(self):
        text = 'Sure! {"action_type": "order", "product_id": "food", "quantity": 300}'
        action = self.inf.extract_action(text)
        assert action["action_type"] == "order"

    def test_extract_action_invalid_json_returns_hold(self):
        action = self.inf.extract_action("I cannot decide right now.")
        assert action["action_type"] == "hold"

    def test_extract_action_unknown_type_returns_hold(self):
        action = self.inf.extract_action('{"action_type": "unknown_xyz"}')
        assert action["action_type"] == "hold"

    def test_build_prompt_returns_two_strings(self):
        system, user = self.inf.build_prompt("inventory_management", GOOD_OBS, 0)
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and len(user) > 0

    def test_build_prompt_all_task_ids(self):
        for task_id in ["inventory_management", "supplier_negotiation", "disruption_response"]:
            system, user = self.inf.build_prompt(task_id, GOOD_OBS, 0)
            assert "action_type" in system  # prompt must mention the schema


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Local Server Smoke Test
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.server
class TestPhase3LocalServer:
    """
    Phase 3: Hit all FastAPI endpoints.
    REQUIRES: uvicorn server.app:app --port 8000 to be running.

    Skip automatically if server is not up:
        pytest tests/test_env.py -m "not server"
    """

    @pytest.fixture(autouse=True)
    def require_server(self):
        if not _server_is_up():
            pytest.skip("Local server not running on :7860 — run: uvicorn server.app:app --port 7860")

    def test_health_returns_ok(self):
        resp = _http_get("/health")
        assert resp.get("status") == "ok"

    def test_tasks_returns_three_tasks(self):
        resp = _http_get("/tasks")
        assert isinstance(resp, list)
        assert len(resp) == 3
        ids = {t["task_id"] for t in resp}
        assert ids == {"inventory_management", "supplier_negotiation", "disruption_response"}

    def test_tasks_have_required_fields(self):
        resp = _http_get("/tasks")
        for task in resp:
            for field in ["task_id", "difficulty", "goal", "max_steps", "initial_cash"]:
                assert field in task, f"Missing field '{field}' in task response"

    def test_baseline_returns_three_scores(self):
        resp = _http_get("/baseline")
        assert isinstance(resp, dict)
        assert set(resp.keys()) == {"inventory_management", "supplier_negotiation", "disruption_response"}
        for tid, score in resp.items():
            assert isinstance(score, (int, float)), f"{tid} score must be numeric"

    def test_reset_inventory_management(self):
        resp = _http_post("/reset", {"episode_id": "inventory_management", "seed": 42})
        assert "observation" in resp, f"Reset response missing 'observation': {resp}"

    def test_reset_supplier_negotiation(self):
        resp = _http_post("/reset", {"episode_id": "supplier_negotiation", "seed": 42})
        assert "observation" in resp

    def test_reset_disruption_response(self):
        resp = _http_post("/reset", {"episode_id": "disruption_response", "seed": 42})
        assert "observation" in resp

    def test_step_after_reset_hold_action(self):
        _http_post("/reset", {"episode_id": "inventory_management", "seed": 42})
        resp = _http_post("/step", {"action": {"action_type": "hold"}})
        assert "observation" in resp, f"Step response missing 'observation': {resp}"
        assert "done" in resp

    def test_step_after_reset_order_action(self):
        _http_post("/reset", {"episode_id": "inventory_management", "seed": 42})
        resp = _http_post("/step", {
            "action": {
                "action_type": "order",
                "product_id": "electronics",
                "supplier_id": "supplier_A",
                "quantity": 500,
            }
        })
        assert "observation" in resp

    def test_grade_endpoint_returns_score(self):
        resp = _http_post(
            "/grade/inventory_management",
            [GOOD_OBS] * 5,
        )
        assert "score" in resp
        assert 0.0 <= resp["score"] <= 1.0

    def test_grade_all_three_tasks(self):
        for task_id in ["inventory_management", "supplier_negotiation", "disruption_response"]:
            resp = _http_post(f"/grade/{task_id}", [GOOD_OBS] * 5)
            assert "score" in resp, f"{task_id}: grade response missing 'score'"
            assert 0.0 <= resp["score"] <= 1.0, f"{task_id}: score out of range"

    def test_grade_unknown_task_returns_404(self):
        try:
            _http_post("/grade/nonexistent_task", [GOOD_OBS])
            assert False, "Expected HTTPError 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_observation_fields_after_reset(self):
        resp = _http_post("/reset", {"episode_id": "inventory_management", "seed": 42})
        obs = resp["observation"]
        expected_fields = [
            "inventory_levels", "supplier_reliability",
            "service_level", "cash_balance", "period",
            "disruption_active",
        ]
        for f in expected_fields:
            assert f in obs, f"Observation missing field: {f}"

    def test_full_episode_loop(self):
        """Run a 3-step episode and grade it — end-to-end local test."""
        _http_post("/reset", {"episode_id": "inventory_management", "seed": 42})
        trajectory = []
        for _ in range(3):
            step_resp = _http_post("/step", {"action": {"action_type": "hold"}})
            assert "observation" in step_resp
            trajectory.append(step_resp["observation"])
            if step_resp.get("done"):
                break

        grade_resp = _http_post("/grade/inventory_management", trajectory)
        assert "score" in grade_resp
        assert 0.0 <= grade_resp["score"] <= 1.0
