"""Supply chain environment with sync step(), reset(), and state property."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np

from openenv.core.env_server import Environment
from server.dynamics import BASE_DEMAND, SupplyChainDynamics
from server.models import SupplyChainAction, SupplyChainObservation

# ── Constants ──────────────────────────────────────────────────────────────
_SEED = 42
_MAX_STEPS = 30
_INITIAL_CASH = 500_000.0
_INITIAL_INVENTORY = 1_000

PRODUCTS: list[str] = [
    "electronics",
    "apparel",
    "food",
    "medical",
    "automotive",
]

SUPPLIERS: list[str] = [
    "supplier_A",
    "supplier_B",
    "supplier_C",
    "supplier_D",
]

_INITIAL_RELIABILITY: dict[str, float] = {
    "supplier_A": 0.95,
    "supplier_B": 0.85,
    "supplier_C": 0.75,
    "supplier_D": 0.90,
}


class SupplyChainEnvironment(Environment):
    """OpenEnv-compatible supply chain simulation environment.

    All public methods (``reset``, ``step``, ``state``) are **async** as
    required by the OpenEnv contract and project rule #8.
    """

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=_SEED)
        self._dynamics = SupplyChainDynamics()
        self._env_state: dict[str, Any] = {}
        self._episode_id: str = ""
        self._step_count: int = 0
        self._max_steps: int = _MAX_STEPS

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplyChainObservation:
        """Initialise a new episode and return the starting observation."""
        effective_seed = seed if seed is not None else _SEED
        self._rng = np.random.RandomState(seed=effective_seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0

        self._env_state = {
            "inventory_levels": {p: _INITIAL_INVENTORY for p in PRODUCTS},
            "pending_orders": [],
            "supplier_reliability": dict(_INITIAL_RELIABILITY),
            "demand_forecast": {
                p: BASE_DEMAND[p] for p in PRODUCTS
            },
            "current_costs": 0.0,
            "service_level": 1.0,
            "period": 0,
            "disruption_active": False,
            "disruption_type": None,
            "cash_balance": _INITIAL_CASH,
            "max_steps": self._max_steps,
            "_period_order_costs": 0.0,
        }

        return self._build_observation(
            message="Episode started. You manage a supply chain with "
            f"{len(PRODUCTS)} products and {len(SUPPLIERS)} suppliers.",
        )

    # ── Step ───────────────────────────────────────────────────────────

    def step(
        self,
        action: SupplyChainAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupplyChainObservation:
        """Execute *action*, advance the simulation, and return the new
        observation."""
        # Convert the Pydantic model to a plain dict for dynamics
        action_dict = action.model_dump()

        # 1. Apply the agent's action
        self._env_state = self._dynamics.apply_action(
            self._env_state, action_dict, self._rng,
        )

        # 2. Advance the period (demand, deliveries, costs)
        self._env_state = self._dynamics.advance_period(
            self._env_state, self._rng,
        )

        self._step_count += 1

        # 3. Check termination
        done = self._dynamics.check_done(self._env_state)

        # 4. Build human-readable message
        msg_parts: list[str] = [
            f"Period {self._env_state['period']}/{self._max_steps}.",
            f"Service level: {self._env_state['service_level']:.1%}.",
            f"Cash: ${self._env_state['cash_balance']:,.0f}.",
            f"Action: {action.action_type}.",
        ]
        if done:
            if self._env_state["cash_balance"] <= 0:
                msg_parts.append("BANKRUPT — episode over.")
            else:
                msg_parts.append("Max steps reached — episode over.")

        obs = self._build_observation(
            message=" ".join(msg_parts),
            done=done,
        )
        return obs

    # ── State ──────────────────────────────────────────────────────────

    @property
    def state(self) -> dict[str, Any]:
        """Return episode metadata."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._dynamics.check_done(self._env_state),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_observation(
        self,
        message: str = "",
        done: bool = False,
    ) -> SupplyChainObservation:
        """Construct a ``SupplyChainObservation`` from internal state."""
        s = self._env_state
        return SupplyChainObservation(
            inventory_levels=dict(s["inventory_levels"]),
            pending_orders=[dict(o) for o in s["pending_orders"]],
            supplier_reliability=dict(s["supplier_reliability"]),
            demand_forecast=dict(s["demand_forecast"]),
            current_costs=s["current_costs"],
            service_level=s["service_level"],
            period=s["period"],
            disruption_active=s["disruption_active"],
            disruption_type=s["disruption_type"],
            cash_balance=s["cash_balance"],
            message=message,
            done=done,
            reward=0.0,
        )

    def close(self) -> None:
        """Clean up the environment resources."""
        pass
