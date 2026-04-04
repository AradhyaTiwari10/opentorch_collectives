"""Supply chain environment with sync step(), reset(), and state property.

State management: module-level singletons are used so that the stateless
HTTP server wrapper (openenv create_fastapi_app) can share episode state
across the lifespan of a single server process.  For multi-tenant / parallel
evaluation, replace with an asyncio-safe session-keyed store.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np

from openenv.core.env_server import Environment
from server.dynamics import BASE_DEMAND, SupplyChainDynamics
from server.events import DisruptionEventEngine
from server.models import SupplyChainAction, SupplyChainObservation
from server.tasks import TASK_REGISTRY

# ── Constants ──────────────────────────────────────────────────────────────
_SEED = 42
_MAX_STEPS_DEFAULT = 30
_INITIAL_CASH = 500_000.0
_INITIAL_INVENTORY = 1_000

# CO2e emission factors (kg per unit)
_CARBON_EMERGENCY = 5.0   # air freight
_CARBON_ORDER = 1.0       # sea / road freight
_CARBON_CEILING = 100_000.0  # normalisation ceiling for sustainability score

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

# Pre-computed starting mean — used by grader to avoid hardcoded literals
INITIAL_MEAN_RELIABILITY: float = (
    sum(_INITIAL_RELIABILITY.values()) / len(_INITIAL_RELIABILITY)
)  # = 0.8625


# ── Module-level singleton state ───────────────────────────────────────────
# Shared across all Environment instances within one server process.
# Safe for sequential evaluation. For concurrent multi-agent use, swap for
# asyncio-safe session-keyed storage (e.g. Redis, or dict keyed by client-id).

_g_dynamics: SupplyChainDynamics = SupplyChainDynamics()
_g_rng: np.random.RandomState = np.random.RandomState(seed=_SEED)
_g_env_state: dict[str, Any] = {}
_g_episode_id: str = ""
_g_step_count: int = 0
_g_max_steps: int = _MAX_STEPS_DEFAULT


class SupplyChainEnvironment(Environment):
    """OpenEnv-compatible supply chain simulation environment.

    Implements the full OpenEnv contract:
      reset(seed, episode_id) → SupplyChainObservation
      step(action)            → SupplyChainObservation
      state                   → dict (episode metadata)
      close()                 → None

    Novel mechanics vs. standard supply chain RL environments:
      - Forced disruption injection (port strikes, supplier failures at
        deterministic periods for reproducible evaluation).
      - Carbon footprint tracking with a sustainability reward component —
        penalises excessive use of high-emission emergency sourcing.
      - Full event lifecycle via DisruptionEventEngine: events expire,
        supplier reliability auto-restores after supplier_failure events.
    """

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplyChainObservation:
        """Initialise a new episode and return the starting observation."""
        global _g_rng, _g_env_state, _g_episode_id, _g_step_count, _g_max_steps

        effective_seed = seed if seed is not None else _SEED
        _g_rng = np.random.RandomState(seed=effective_seed)
        _g_episode_id = episode_id or str(uuid.uuid4())
        _g_step_count = 0

        task_cfg = TASK_REGISTRY.get(_g_episode_id)
        starting_cash = task_cfg.initial_cash if task_cfg else _INITIAL_CASH
        _g_max_steps = task_cfg.max_steps if task_cfg else _MAX_STEPS_DEFAULT

        _g_env_state = {
            "inventory_levels": {p: _INITIAL_INVENTORY for p in PRODUCTS},
            "pending_orders": [],
            "supplier_reliability": dict(_INITIAL_RELIABILITY),
            "demand_forecast": {p: BASE_DEMAND[p] for p in PRODUCTS},
            "current_costs": 0.0,
            "service_level": 1.0,
            "period": 0,
            "disruption_active": False,
            "disruption_type": None,
            "cash_balance": starting_cash,
            "carbon_footprint": 0.0,
            "max_steps": _g_max_steps,
            "_period_order_costs": 0.0,
            "_active_events": {},          # tracks disruption durations
        }

        return self._build_observation(
            message=(
                f"Episode '{_g_episode_id}' started. "
                f"You manage a supply chain with {len(PRODUCTS)} products "
                f"and {len(SUPPLIERS)} suppliers over {_g_max_steps} periods."
            ),
        )

    # ── Step ───────────────────────────────────────────────────────────

    def step(
        self,
        action: SupplyChainAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupplyChainObservation:
        """Execute action, advance the simulation, and return the new observation."""
        global _g_env_state, _g_step_count

        action_dict = action.model_dump()

        # ── 1. Apply agent action ──────────────────────────────────────
        _g_env_state = _g_dynamics.apply_action(_g_env_state, action_dict, _g_rng)

        # ── 2. Carbon footprint tracking ───────────────────────────────
        action_type = action_dict.get("action_type", "hold")
        qty: int = action_dict.get("quantity") or 0
        if action_type == "emergency_source":
            _g_env_state["carbon_footprint"] = (
                _g_env_state.get("carbon_footprint", 0.0) + qty * _CARBON_EMERGENCY
            )
        elif action_type == "order":
            _g_env_state["carbon_footprint"] = (
                _g_env_state.get("carbon_footprint", 0.0) + qty * _CARBON_ORDER
            )

        # ── 3. Inject forced disruptions (before period advance) ───────
        task_cfg = TASK_REGISTRY.get(_g_episode_id)
        if task_cfg and task_cfg.forced_events:
            current_period = _g_env_state.get("period", 0)
            forced_event = task_cfg.forced_events.get(current_period)
            if forced_event:
                _g_env_state = DisruptionEventEngine.apply_event(
                    _g_env_state, forced_event
                )
            elif not task_cfg.disruptions_enabled:
                # Clear disruption state when random events are disabled
                # and no forced event is active this period
                if not _g_env_state.get("_active_events"):
                    _g_env_state["disruption_active"] = False
                    _g_env_state["disruption_type"] = None

        # ── 4. Advance the period (demand, deliveries, costs) ──────────
        _g_env_state = _g_dynamics.advance_period(_g_env_state, _g_rng)

        # ── 5. Tick event lifecycle — expire finished disruptions ───────
        _g_env_state = DisruptionEventEngine.tick_events(_g_env_state)

        # ── 6. Sample random events for tasks that allow them ──────────
        if task_cfg:
            prob = getattr(task_cfg, "event_probability_override", 0.0)
            advanced_period = _g_env_state.get("period", 0)
            forced_now = (
                task_cfg.forced_events
                and advanced_period in task_cfg.forced_events
            )
            if task_cfg.disruptions_enabled and prob > 0 and not forced_now:
                random_event = DisruptionEventEngine.sample_event(
                    _g_rng, advanced_period
                )
                if random_event is not None:
                    _g_env_state = DisruptionEventEngine.apply_event(
                        _g_env_state, random_event
                    )

        _g_step_count += 1

        # ── 7. Check termination ───────────────────────────────────────
        done = _g_dynamics.check_done(_g_env_state)

        # ── 8. Build human-readable message ───────────────────────────
        carbon = _g_env_state.get("carbon_footprint", 0.0)
        msg_parts: list[str] = [
            f"Period {_g_env_state['period']}/{_g_max_steps}.",
            f"Service: {_g_env_state['service_level']:.1%}.",
            f"Cash: ${_g_env_state['cash_balance']:,.0f}.",
            f"Carbon: {carbon:.0f} kg CO2e.",
            f"Action: {action.action_type}.",
        ]
        if done:
            if _g_env_state["cash_balance"] <= 0:
                msg_parts.append("BANKRUPT — episode over.")
            else:
                msg_parts.append("Max steps reached — episode over.")

        return self._build_observation(
            message=" ".join(msg_parts),
            done=done,
            action=action,
        )

    # ── State ──────────────────────────────────────────────────────────

    @property
    def state(self) -> dict[str, Any]:
        """Return episode metadata (read-only snapshot)."""
        return {
            "episode_id": _g_episode_id,
            "step_count": _g_step_count,
            "max_steps": _g_max_steps,
            "done": _g_dynamics.check_done(_g_env_state),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_observation(
        self,
        message: str = "",
        done: bool = False,
        action: Optional[SupplyChainAction] = None,
    ) -> SupplyChainObservation:
        """Construct a SupplyChainObservation from internal state."""
        s = _g_env_state

        # cost_score: 1.0 when period cost = 0, 0.0 when cost >= 50k
        cost_score = max(0.0, min(1.0, 1.0 - (s["current_costs"] / 50_000.0)))

        # service_score: fraction of demand fulfilled this period
        service_score = max(0.0, min(1.0, float(s["service_level"])))

        # resilience_score: rewards active crisis management over holding
        resilience_score = 1.0
        if s.get("disruption_active"):
            if action and action.action_type in ["emergency_source", "reroute"]:
                resilience_score = 0.8   # proactive response
            elif action and action.action_type == "hold":
                resilience_score = 0.2   # passive — worst during disruption
            else:
                resilience_score = 0.5   # ordering / negotiating

        # sustainability_score: penalises cumulative carbon footprint
        carbon = s.get("carbon_footprint", 0.0)
        sustainability_score = max(0.0, min(1.0, 1.0 - (carbon / _CARBON_CEILING)))

        # Multi-objective reward (weights sum to 1.0)
        total_reward = (
            0.35 * cost_score
            + 0.35 * service_score
            + 0.15 * resilience_score
            + 0.15 * sustainability_score
        )

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
            carbon_footprint=carbon,
            message=message,
            done=done,
            reward=total_reward,
        )

    def close(self) -> None:
        """Clean up the environment resources."""
        pass
