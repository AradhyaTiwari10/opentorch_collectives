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

# ── Dynamic starting reliability ──────────────────────────────────────────
# Imported from environment so grader stays consistent with simulation.
# Avoids hardcoded literals that drift when _INITIAL_RELIABILITY changes.
from server.environment import INITIAL_MEAN_RELIABILITY as _START_RELIABILITY


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
    """EASY — score = longest consecutive streak of periods with service_level >= threshold,
    normalised by the target number of consecutive periods.

    Uses *consecutive* periods as specified in the task goal, not a simple total count.
    Partial credit is awarded proportional to the best streak achieved.
    """

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

    # Track longest consecutive streak above threshold
    longest_streak = 0
    current_streak = 0
    total_above = 0   # kept for informational details

    for step in trajectory:
        if float(step.get("service_level", 0.0)) >= threshold:
            current_streak += 1
            total_above += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            current_streak = 0

    # Score: fraction of the required streak actually achieved (clamped to 1.0)
    score = _clamp(longest_streak / target_periods)
    passed = longest_streak >= target_periods

    return TaskResult(
        task_id="inventory_management",
        score=score,
        passed=passed,
        details={
            "longest_streak": longest_streak,
            "periods_above_threshold": total_above,   # kept for backward compat
            "total_periods": len(trajectory),
            "target_periods": target_periods,
            "threshold": threshold,
        },
    )


def _grade_supplier_negotiation(
    trajectory: list[dict[str, Any]],
) -> TaskResult:
    """MEDIUM — score = 0.70 * rel_score + 0.30 * cash_score, clamped 0-1.

    rel_score: improvement from baseline mean reliability toward the target.
    cash_score: partial credit for maintaining positive cash balance.

    Baseline uses the dynamically computed INITIAL_MEAN_RELIABILITY so the
    grader stays consistent with the simulation even if constants change.
    """

    task_cfg = TASK_REGISTRY["supplier_negotiation"]
    target_reliability: float = task_cfg.success_criteria["reliability_target"]
    min_cash: float = task_cfg.success_criteria["min_cash_balance"]

    if not trajectory:
        return TaskResult(
            task_id="supplier_negotiation",
            score=0.0,
            passed=False,
            details={"error": "empty trajectory"},
        )

    final_step = trajectory[-1]

    # Mean supplier reliability at end of episode
    reliabilities: dict[str, float] = final_step.get("supplier_reliability", {})
    mean_reliability = (
        sum(reliabilities.values()) / len(reliabilities)
        if reliabilities
        else 0.0
    )

    cash_balance: float = float(final_step.get("cash_balance", 0.0))

    # Reliability score: only award credit for improvement above baseline.
    # A hold agent that never negotiates scores exactly 0.0.
    if mean_reliability <= _START_RELIABILITY:
        rel_score = 0.0
    else:
        rel_score = _clamp(
            (mean_reliability - _START_RELIABILITY)
            / (target_reliability - _START_RELIABILITY)
        )

    # Cash score: partial credit as long as agent didn't go bankrupt
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
            "start_reliability": round(_START_RELIABILITY, 4),
            "reliability_target": target_reliability,
            "cash_balance": round(cash_balance, 2),
            "min_cash_required": min_cash,
        },
    )


def _grade_disruption_response(
    trajectory: list[dict[str, Any]],
) -> TaskResult:
    """HARD — score = average recovery speed across both forced disruptions.

    For each disruption, measures the fraction of the recovery window
    (next N periods after the disruption) where service_level >= threshold.
    An agent that recovers immediately scores 1.0 for that disruption;
    one that never recovers scores 0.0.
    """

    task_cfg = TASK_REGISTRY["disruption_response"]
    threshold: float = task_cfg.success_criteria["service_level_threshold"]
    window: int = task_cfg.success_criteria["recovery_window"]
    disruption_periods: list[int] = task_cfg.success_criteria["disruption_periods"]

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
        # Find observations that fall within the recovery window
        post_disruption = [
            step for step in trajectory
            if d_period < step.get("period", -1) <= d_period + window
        ]

        if not post_disruption:
            recovery_scores.append(0.0)
            all_recovered = False
            continue

        periods_above = sum(
            1
            for step in post_disruption
            if float(step.get("service_level", 0.0)) >= threshold
        )

        # Recovery speed = fraction of window that met the threshold
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
    """Score a single task. Returns a TaskResult with score in 0.0-1.0.

    Pure function — no side-effects, no global state. Fully deterministic.
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

    Returns a dict mapping task_id → TaskResult.
    Sorted iteration ensures fully reproducible output order.
    """
    return {
        task_id: grade_task(task_id, trajectories[task_id])
        for task_id in sorted(trajectories)
    }
