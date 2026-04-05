"""Reward computation for the supply chain environment.

Canonical source for reward weights and component calculations.
Used by environment.py via RewardEngine.compute_reward().

Reward formula (weights sum to 1.0):
    total_reward = 0.35 * cost_score
                 + 0.35 * service_score
                 + 0.15 * resilience_score
                 + 0.15 * sustainability_score
"""

from __future__ import annotations

import math
from typing import Any

from server.models import SupplyChainReward

# ── Baseline constants ────────────────────────────────────────────────────
COST_BASELINE: float = 50_000.0
CARBON_CEILING: float = 100_000.0  # normalisation ceiling for sustainability

# ── Reward weights (must sum to 1.0) ──────────────────────────────────────
W_COST: float = 0.35
W_SERVICE: float = 0.35
W_RESILIENCE: float = 0.15
W_SUSTAINABILITY: float = 0.15


def _safe_float(value: float) -> float:
    """Return *value* clamped to [0, 1], replacing NaN / Inf with 0."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


class RewardEngine:
    """Compute a multi-objective :class:`SupplyChainReward` each step.

    All component scores are clamped to [0, 1] and the ``total_reward``
    is guaranteed to be a finite float in [0, 1].

    Four components:
      - **cost_score**: penalises high period costs
      - **service_score**: fraction of demand fulfilled
      - **resilience_score**: rewards active crisis management
      - **sustainability_score**: penalises cumulative carbon footprint
    """

    @staticmethod
    def compute_reward(
        env_state: dict[str, Any],
        action_type: str = "hold",
    ) -> SupplyChainReward:
        """Return the reward for the current environment state.

        Args:
            env_state: current environment state dictionary.
            action_type: the action_type string taken this step.

        Returns:
            SupplyChainReward with all four component scores and total.
        """

        # ── cost_score ────────────────────────────────────────────────
        period_cost: float = float(env_state.get("current_costs", 0.0))
        cost_score = _safe_float(1.0 - (period_cost / COST_BASELINE))

        # ── service_score ─────────────────────────────────────────────
        service_score = _safe_float(float(env_state.get("service_level", 1.0)))

        # ── resilience_score ──────────────────────────────────────────
        disruption_active = env_state.get("disruption_active", False)

        if not disruption_active:
            resilience_score = 1.0
        elif action_type in ("emergency_source", "reroute"):
            resilience_score = 0.8   # proactive response
        elif action_type == "hold":
            resilience_score = 0.2   # passive — worst during disruption
        else:
            resilience_score = 0.5   # ordering / negotiating

        # ── sustainability_score ──────────────────────────────────────
        carbon: float = float(env_state.get("carbon_footprint", 0.0))
        sustainability_score = _safe_float(1.0 - (carbon / CARBON_CEILING))

        # ── total_reward (weighted sum) ───────────────────────────────
        total_reward = (
            W_COST * cost_score
            + W_SERVICE * service_score
            + W_RESILIENCE * resilience_score
            + W_SUSTAINABILITY * sustainability_score
        )

        return SupplyChainReward(
            cost_score=_safe_float(cost_score),
            service_score=_safe_float(service_score),
            resilience_score=_safe_float(resilience_score),
            sustainability_score=_safe_float(sustainability_score),
            total_reward=_safe_float(total_reward),
        )
