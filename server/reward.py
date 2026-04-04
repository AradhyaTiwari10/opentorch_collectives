"""Reward computation for the supply chain environment."""

from __future__ import annotations

import math
from typing import Any

from server.events import DisruptionEventEngine
from server.models import SupplyChainReward

# ── Baseline constant ─────────────────────────────────────────────────────
COST_BASELINE: float = 50_000.0


def _safe_float(value: float) -> float:
    """Return *value* clamped to [-1, 1], replacing NaN / Inf with 0."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(-1.0, min(1.0, value))


class RewardEngine:
    """Compute a multi-objective :class:`SupplyChainReward` each step.

    All component scores are clamped to [0, 1] and the ``total_reward``
    is guaranteed to be a finite float in [-1, 1].
    """

    @staticmethod
    def compute_reward(
        env_state: dict[str, Any],
        action: dict[str, Any],
        next_env_state: dict[str, Any],
    ) -> SupplyChainReward:
        """Return the reward for transitioning from *env_state* to
        *next_env_state* via *action*."""

        # ── cost_score ────────────────────────────────────────────────
        period_cost: float = float(next_env_state.get("current_costs", 0.0))
        cost_score = max(0.0, min(1.0, 1.0 - (period_cost / COST_BASELINE)))

        # ── service_score ─────────────────────────────────────────────
        service_score = max(
            0.0,
            min(1.0, float(next_env_state.get("service_level", 1.0))),
        )

        # ── resilience_score ──────────────────────────────────────────
        disruption_active = any(
            DisruptionEventEngine.is_event_active(next_env_state, et)
            for et in ("port_strike", "supplier_failure", "demand_surge")
        )

        if not disruption_active:
            resilience_score = 1.0
        else:
            action_type = action.get("action_type", "hold")
            if action_type in ("reroute", "emergency_source"):
                resilience_score = 0.8
            elif action_type in ("order", "negotiate"):
                resilience_score = 0.5
            else:
                # "hold" or any unknown action during disruption
                resilience_score = 0.2

        # ── total_reward ──────────────────────────────────────────────
        total_reward = (
            0.4 * cost_score
            + 0.4 * service_score
            + 0.2 * resilience_score
        )

        return SupplyChainReward(
            cost_score=_safe_float(cost_score),
            service_score=_safe_float(service_score),
            resilience_score=_safe_float(resilience_score),
            total_reward=_safe_float(total_reward),
        )
