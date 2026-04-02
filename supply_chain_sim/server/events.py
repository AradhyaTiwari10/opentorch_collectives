"""Random event generation for supply chain disruptions and market changes."""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Event configuration ───────────────────────────────────────────────────
# Probabilities must sum to 1.0 with "no event" as the complement.
_EVENT_PROBABILITIES: dict[str | None, float] = {
    "port_strike": 0.05,
    "supplier_failure": 0.08,
    "demand_surge": 0.10,
    None: 0.77,  # no event
}

# How long each disruption lingers once triggered (periods).
_EVENT_DURATION: dict[str, int] = {
    "port_strike": 1,        # one-shot delay applied immediately
    "supplier_failure": 3,   # reliability stays degraded for 3 periods
    "demand_surge": 2,       # doubled demand for 2 periods
}

# Products affected by a port strike
_PORT_STRIKE_PRODUCTS: list[str] = ["apparel", "electronics"]
_PORT_STRIKE_DELAY_DAYS = 5
_PORT_STRIKE_COST_MULTIPLIER = 1.20  # +20 %

# Supplier failure drops reliability to this floor
_SUPPLIER_FAILURE_RELIABILITY = 0.1

# Demand surge multiplier
_DEMAND_SURGE_MULTIPLIER = 2.0


class DisruptionEventEngine:
    """Probabilistic disruption events that fire each period.

    All randomness is drawn from the caller-supplied ``rng``
    (``numpy.random.RandomState``), never from global random state.
    """

    # ── Sample ────────────────────────────────────────────────────────

    @staticmethod
    def sample_event(
        rng: np.random.RandomState,
        period: int,  # noqa: ARG004 — kept for future period-dependent logic
    ) -> str | None:
        """Return an event type or ``None`` (no event) for this period."""
        events = list(_EVENT_PROBABILITIES.keys())
        probabilities = [_EVENT_PROBABILITIES[e] for e in events]
        idx: int = int(rng.choice(len(events), p=probabilities))
        return events[idx]

    # ── Apply ─────────────────────────────────────────────────────────

    @staticmethod
    def apply_event(
        env_state: dict[str, Any],
        event_type: str | None,
    ) -> dict[str, Any]:
        """Mutate *env_state* with the effects of *event_type* and return it."""
        if event_type is None:
            env_state["disruption_active"] = False
            env_state["disruption_type"] = None
            return env_state

        env_state["disruption_active"] = True
        env_state["disruption_type"] = event_type

        # Track remaining duration so downstream code can expire events.
        active_events: dict[str, dict[str, Any]] = env_state.setdefault(
            "_active_events", {},
        )

        if event_type == "port_strike":
            active_events["port_strike"] = {
                "remaining": _EVENT_DURATION["port_strike"],
            }
            # Delay all pending orders for affected products by +5 days
            for order in env_state.get("pending_orders", []):
                if order.get("product") in _PORT_STRIKE_PRODUCTS:
                    order["eta_days"] += _PORT_STRIKE_DELAY_DAYS

            # Increase period order costs by 20 %
            current_order_costs = env_state.get("_period_order_costs", 0.0)
            env_state["_period_order_costs"] = (
                current_order_costs * _PORT_STRIKE_COST_MULTIPLIER
            )

        elif event_type == "supplier_failure":
            # Pick a random supplier to fail (use stored rng via env_state)
            suppliers = list(env_state.get("supplier_reliability", {}).keys())
            if suppliers:
                # Deterministic choice: use hash of period as a simple
                # deterministic selector so we don't need rng here.
                failed_supplier = suppliers[
                    env_state.get("period", 0) % len(suppliers)
                ]
                original_reliability = env_state["supplier_reliability"].get(
                    failed_supplier, 0.80,
                )
                active_events["supplier_failure"] = {
                    "remaining": _EVENT_DURATION["supplier_failure"],
                    "supplier": failed_supplier,
                    "original_reliability": original_reliability,
                }
                env_state["supplier_reliability"][failed_supplier] = (
                    _SUPPLIER_FAILURE_RELIABILITY
                )

        elif event_type == "demand_surge":
            # Pick a random product (deterministic via period)
            products = list(env_state.get("inventory_levels", {}).keys())
            if products:
                surged_product = products[
                    env_state.get("period", 0) % len(products)
                ]
                active_events["demand_surge"] = {
                    "remaining": _EVENT_DURATION["demand_surge"],
                    "product": surged_product,
                    "multiplier": _DEMAND_SURGE_MULTIPLIER,
                }
                # Double the demand forecast for this product
                forecast = env_state.get("demand_forecast", {})
                if surged_product in forecast:
                    forecast[surged_product] = int(
                        forecast[surged_product] * _DEMAND_SURGE_MULTIPLIER,
                    )

        return env_state

    # ── Query ─────────────────────────────────────────────────────────

    @staticmethod
    def is_event_active(
        env_state: dict[str, Any],
        event_type: str,
    ) -> bool:
        """Return ``True`` if *event_type* is currently active."""
        active_events: dict[str, dict[str, Any]] = env_state.get(
            "_active_events", {},
        )
        info = active_events.get(event_type)
        if info is None:
            return False
        return info.get("remaining", 0) > 0

    # ── Tick (call once per period to expire events) ──────────────────

    @staticmethod
    def tick_events(env_state: dict[str, Any]) -> dict[str, Any]:
        """Decrement remaining durations and expire finished events."""
        active_events: dict[str, dict[str, Any]] = env_state.get(
            "_active_events", {},
        )
        expired: list[str] = []
        for etype, info in active_events.items():
            info["remaining"] -= 1
            if info["remaining"] <= 0:
                expired.append(etype)
                # Restore side-effects on expiry
                if etype == "supplier_failure":
                    supplier = info.get("supplier")
                    original = info.get("original_reliability", 0.80)
                    if supplier:
                        env_state["supplier_reliability"][supplier] = original

        for etype in expired:
            del active_events[etype]

        # Update top-level flags
        if not active_events:
            env_state["disruption_active"] = False
            env_state["disruption_type"] = None
        else:
            # Keep the most recently added active event as the reported type
            env_state["disruption_type"] = next(iter(active_events))

        return env_state
