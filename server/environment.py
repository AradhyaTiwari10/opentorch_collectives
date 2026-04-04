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
    
    _global_rng = np.random.RandomState(seed=_SEED)
    _global_dynamics = SupplyChainDynamics()
    _global_env_state: dict[str, Any] = {}
    _global_episode_id: str = ""
    _global_step_count: int = 0
    _global_max_steps: int = _MAX_STEPS

    def __init__(self) -> None:
        self._rng = SupplyChainEnvironment._global_rng
        self._dynamics = SupplyChainEnvironment._global_dynamics
        self._env_state = SupplyChainEnvironment._global_env_state
        self._episode_id = SupplyChainEnvironment._global_episode_id
        self._step_count = SupplyChainEnvironment._global_step_count
        self._max_steps = SupplyChainEnvironment._global_max_steps

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

        from server.tasks import TASK_REGISTRY
        task_cfg = TASK_REGISTRY.get(self._episode_id)
        
        starting_cash = task_cfg.initial_cash if task_cfg else _INITIAL_CASH
        self._max_steps = task_cfg.max_steps if task_cfg else _MAX_STEPS

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
            "cash_balance": starting_cash,
            "max_steps": self._max_steps,
            "_period_order_costs": 0.0,
        }

        # Sync class variables for stateless HTTP server wrapper workaround
        SupplyChainEnvironment._global_env_state = self._env_state
        SupplyChainEnvironment._global_episode_id = self._episode_id
        SupplyChainEnvironment._global_step_count = self._step_count
        SupplyChainEnvironment._global_max_steps = self._max_steps

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

        # 1b. Inject forced disruptions from task config (before period advance)
        from server.tasks import TASK_REGISTRY as _TR
        _task_cfg = _TR.get(self._episode_id)
        if _task_cfg and _task_cfg.forced_events:
            _current_period = self._env_state.get("period", 0)
            _event = _task_cfg.forced_events.get(_current_period)
            if _event:
                self._env_state["disruption_active"] = True
                self._env_state["disruption_type"] = _event
            elif not _task_cfg.disruptions_enabled:
                # Clear disruption when no random events enabled
                self._env_state["disruption_active"] = False
                self._env_state["disruption_type"] = None

        # 2. Advance the period (demand, deliveries, costs)
        self._env_state = self._dynamics.advance_period(
            self._env_state, self._rng,
        )

        self._step_count += 1
        
        # Sync class variables for stateless HTTP server wrapper workaround
        SupplyChainEnvironment._global_env_state = self._env_state
        SupplyChainEnvironment._global_step_count = self._step_count

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
            action=action,
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
        action: Optional[SupplyChainAction] = None,
    ) -> SupplyChainObservation:
        """Construct a ``SupplyChainObservation`` from internal state."""
        s = self._env_state

        # Calculate step reward
        cost_score = max(0.0, min(1.0, 1.0 - (s["current_costs"] / 50000.0)))
        service_score = max(0.0, min(1.0, float(s["service_level"])))
        
        resilience_score = 1.0
        if s.get("disruption_active"):
            if action and action.action_type in ["emergency_source", "reroute"]:
                resilience_score = 0.8
            elif action and action.action_type == "hold":
                resilience_score = 0.2
            else:
                resilience_score = 0.5
                
        total_reward = (0.4 * cost_score) + (0.4 * service_score) + (0.2 * resilience_score)

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
            reward=total_reward,
        )

    def close(self) -> None:
        """Clean up the environment resources."""
        pass
