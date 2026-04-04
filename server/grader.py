"""Grading logic for supply chain tasks.

Grader scores must always be a float between 0.0 and 1.0.

All functions are **pure** — no side-effects, no global state, and fully
reproducible with seed=42.
"""

from __future__ import annotations

import math
from typing import Any

from server.models import TaskResult
from server.tasks import TASK_REGISTRY

# ── Seed (used only if a grader ever needs randomness — currently none) ───
_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi], replacing NaN/Inf with 0."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(lo, min(hi, value))


# ── Per-task graders ──────────────────────────────────────────────────────

def _grade_inventory_management(
    trajectory: list[dict[str, Any]],
) -> TaskResult:
    """EASY — score = fraction of all periods with service_level >= threshold."""

    task_cfg = TASK_REGISTRY["inventory_management"]
    threshold: float = task_cfg.success_criteria["service_level_threshold"]
    target_periods: int = task_cfg.success_criteria["consecutive_periods"]

    if not trajectory:
        return TaskResult(
            task_id="inventory_management",
            score=0.0,
            passed=False,
            details={"error": "empty trajectory"},
        )

    # Score over ALL periods (not just last 10) for a fairer metric
    periods_above = sum(
        1
        for step in trajectory
        if float(step.get("service_level", 0.0)) >= threshold
    )

    score = _clamp(periods_above / max(len(trajectory), target_periods))
    passed = periods_above >= target_periods

    return TaskResult(
        task_id="inventory_management",
        score=score,
        passed=passed,
        details={
            "periods_above_threshold": periods_above,
            "total_periods": len(trajectory),
            "target_periods": target_periods,
            "threshold": threshold,
        },
    )


def _grade_supplier_negotiation(
    trajectory: list[dict[str, Any]],
) -> TaskResult:
    """MEDIUM — score = (reliability / 0.92) × (cash / 300k), clamped 0-1."""

    task_cfg = TASK_REGISTRY["supplier_negotiation"]
    target_reliability: float = task_cfg.success_criteria["reliability_target"]
    min_cash: float = task_cfg.success_criteria["min_cash_balance"]
    initial_cash: float = task_cfg.initial_cash

    if not trajectory:
        return TaskResult(
            task_id="supplier_negotiation",
            score=0.0,
            passed=False,
            details={"error": "empty trajectory"},
        )

    final_step = trajectory[-1]

    # Mean supplier reliability at end of episode
    reliabilities: dict[str, float] = final_step.get(
        "supplier_reliability", {},
    )
    if reliabilities:
        mean_reliability = sum(reliabilities.values()) / len(reliabilities)
    else:
        mean_reliability = 0.0

    cash_balance: float = float(final_step.get("cash_balance", 0.0))

    # Delta logic: ensure 'hold' completely fails (score 0) since initial is 0.86
    start_reliability = 0.86  # mean of _INITIAL_RELIABILITY
    if mean_reliability <= start_reliability:
        rel_score = 0.0
    else:
        rel_score = _clamp(
            (mean_reliability - start_reliability) / (target_reliability - start_reliability)
        )

    # Cash score: partial credit as long as agent didn't go deeply negative
    # 1.0 if cash >= min_cash, 0.5 if at break-even, 0.0 if bankrupt
    if cash_balance >= min_cash:
        cash_score = 1.0
    elif cash_balance > 0:
        cash_score = cash_balance / min_cash
    else:
        cash_score = 0.0

    # Weighted combination: reliability matters more (70%) than cash (30%)
    score = _clamp(0.70 * rel_score + 0.30 * cash_score)

    passed = mean_reliability >= target_reliability and cash_balance > min_cash

    return TaskResult(
        task_id="supplier_negotiation",
        score=score,
        passed=passed,
        details={
            "mean_reliability": round(mean_reliability, 4),
            "reliability_target": target_reliability,
            "cash_balance": round(cash_balance, 2),
            "min_cash_required": min_cash,
        },
    )


def _grade_disruption_response(
    trajectory: list[dict[str, Any]],
) -> TaskResult:
    """HARD — score = average recovery speed across both disruptions."""

    task_cfg = TASK_REGISTRY["disruption_response"]
    threshold: float = task_cfg.success_criteria["service_level_threshold"]
    window: int = task_cfg.success_criteria["recovery_window"]
    disruption_periods: list[int] = task_cfg.success_criteria[
        "disruption_periods"
    ]

    if not trajectory:
        return TaskResult(
            task_id="disruption_response",
            score=0.0,
            passed=False,
            details={"error": "empty trajectory"},
        )

    recovery_scores: list[float] = []
    all_recovered = True

    for d_period in disruption_periods:
        # Find steps that fall within the recovery window after the disruption
        # Match by the 'period' field in each step (period is 1-indexed after advance)
        post_disruption = [
            step for step in trajectory
            if d_period < step.get("period", -1) <= d_period + window
        ]

        if not post_disruption:
            recovery_scores.append(0.0)
            all_recovered = False
            continue

        # Count how many periods in the recovery window met the threshold
        periods_above = sum(
            1
            for step in post_disruption
            if float(step.get("service_level", 0.0)) >= threshold
        )

        # recovery_speed = fraction of the window that was above threshold
        recovery_speed = periods_above / window if window else 0.0
        recovery_scores.append(_clamp(recovery_speed))

        if periods_above < window:
            all_recovered = False

    avg_recovery = (
        sum(recovery_scores) / len(recovery_scores) if recovery_scores else 0.0
    )
    score = _clamp(avg_recovery)

    return TaskResult(
        task_id="disruption_response",
        score=score,
        passed=all_recovered,
        details={
            "recovery_scores": [round(s, 4) for s in recovery_scores],
            "disruption_periods": disruption_periods,
            "window": window,
            "threshold": threshold,
        },
    )


# ── Dispatcher ─────────────────────────────────────────────────────────────

_GRADER_MAP: dict[str, Any] = {
    "inventory_management": _grade_inventory_management,
    "supplier_negotiation": _grade_supplier_negotiation,
    "disruption_response": _grade_disruption_response,
}


def grade_task(
    task_id: str,
    episode_trajectory: list[dict[str, Any]],
) -> TaskResult:
    """Score a single task. Returns a :class:`TaskResult` with score 0.0-1.0.

    Pure function — no side-effects, no global state.
    """
    grader = _GRADER_MAP.get(task_id)
    if grader is None:
        return TaskResult(
            task_id=task_id,
            score=0.0,
            passed=False,
            details={"error": f"Unknown task_id: {task_id!r}"},
        )
    return grader(episode_trajectory)


def run_all_graders(
    trajectories: dict[str, list[dict[str, Any]]],
) -> dict[str, TaskResult]:
    """Grade every task whose trajectory is provided.

    Returns a dict mapping ``task_id → TaskResult``.
    Fully reproducible with seed=42 (all graders are deterministic).
    """
    results: dict[str, TaskResult] = {}
    for task_id in sorted(trajectories):  # sorted for determinism
        results[task_id] = grade_task(task_id, trajectories[task_id])
    return results
