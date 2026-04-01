"""Supply chain dynamics — demand, lead times, inventory flow, and logistics."""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Cost constants ─────────────────────────────────────────────────────────
UNIT_ORDER_COST = 10.0            # base cost per unit ordered
HOLDING_COST_PER_UNIT = 0.5       # per-unit per-period inventory carrying cost
STOCKOUT_PENALTY_PER_UNIT = 20.0  # penalty per unit of unmet demand
NEGOTIATION_FEE = 1_000.0         # flat fee for a negotiate action
REROUTE_FEE = 2_000.0             # flat fee for rerouting an order
EMERGENCY_COST_MULTIPLIER = 2.0   # emergency sourcing is 2× normal cost
HOLD_PENALTY = -5.0               # explicit penalty for doing nothing

# Base demand per product per period
BASE_DEMAND: dict[str, int] = {
    "electronics": 200,
    "apparel": 150,
    "food": 300,
    "medical": 100,
    "automotive": 80,
}


class SupplyChainDynamics:
    """Pure-function style dynamics for the supply chain simulation.

    Every method receives the env_state dict and an ``rng``
    (``numpy.random.RandomState``), mutates the state in-place, and
    returns it for convenience.  No global random state is ever used.
    """

    # ── Action handlers ────────────────────────────────────────────────

    @staticmethod
    def apply_action(
        env_state: dict[str, Any],
        action: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Dispatch *action* and apply its effects to *env_state*."""

        action_type: str = action["action_type"]

        if action_type == "order":
            env_state = SupplyChainDynamics._handle_order(env_state, action, rng)
        elif action_type == "negotiate":
            env_state = SupplyChainDynamics._handle_negotiate(env_state, action, rng)
        elif action_type == "reroute":
            env_state = SupplyChainDynamics._handle_reroute(env_state, action, rng)
        elif action_type == "hold":
            env_state = SupplyChainDynamics._handle_hold(env_state)
        elif action_type == "emergency_source":
            env_state = SupplyChainDynamics._handle_emergency_source(
                env_state, action, rng,
            )
        else:
            raise ValueError(f"Unknown action_type: {action_type!r}")

        return env_state

    # ── Period advancement ─────────────────────────────────────────────

    @staticmethod
    def advance_period(
        env_state: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Advance the simulation by one period.

        1. Generate noisy demand and consume inventory.
        2. Deliver orders whose ETA has reached zero.
        3. Compute service level = fulfilled / total demand.
        4. Compute period costs and deduct from cash balance.
        """

        inventory: dict[str, int] = env_state["inventory_levels"]
        pending: list[dict] = env_state["pending_orders"]

        total_demand = 0
        fulfilled_demand = 0
        period_order_costs: float = env_state.get("_period_order_costs", 0.0)
        stockout_cost = 0.0

        # --- 1. Demand consumption (±20 % noise) -----------------------
        for product, base in BASE_DEMAND.items():
            noise_factor = 1.0 + rng.uniform(-0.20, 0.20)
            demand = max(0, int(round(base * noise_factor)))
            total_demand += demand

            available = inventory.get(product, 0)
            fulfilled = min(available, demand)
            fulfilled_demand += fulfilled
            inventory[product] = available - fulfilled

            shortfall = demand - fulfilled
            if shortfall > 0:
                stockout_cost += shortfall * STOCKOUT_PENALTY_PER_UNIT

        # --- 2. Deliver pending orders whose eta <= 0 ------------------
        still_pending: list[dict] = []
        for order in pending:
            order["eta_days"] -= 1
            if order["eta_days"] <= 0:
                prod = order["product"]
                inventory[prod] = inventory.get(prod, 0) + order["qty"]
            else:
                still_pending.append(order)
        env_state["pending_orders"] = still_pending

        # --- 3. Service level ------------------------------------------
        if total_demand > 0:
            env_state["service_level"] = fulfilled_demand / total_demand
        else:
            env_state["service_level"] = 1.0

        # --- 4. Period cost & cash balance -----------------------------
        holding_cost = sum(
            qty * HOLDING_COST_PER_UNIT for qty in inventory.values()
        )
        period_cost = holding_cost + period_order_costs + stockout_cost
        env_state["current_costs"] = period_cost
        env_state["cash_balance"] -= period_cost

        # Update demand forecast for next period (base ± 20 %)
        env_state["demand_forecast"] = {
            prod: max(0, int(round(base * (1.0 + rng.uniform(-0.20, 0.20)))))
            for prod, base in BASE_DEMAND.items()
        }

        # Advance period counter
        env_state["period"] += 1

        # Reset per-period accumulators
        env_state["_period_order_costs"] = 0.0

        return env_state

    # ── Done check ─────────────────────────────────────────────────────

    @staticmethod
    def check_done(env_state: dict[str, Any]) -> bool:
        """Return ``True`` if the episode should end."""
        if env_state["period"] >= env_state["max_steps"]:
            return True
        if env_state["cash_balance"] <= 0:
            return True
        return False

    # ── Private action handlers ────────────────────────────────────────

    @staticmethod
    def _handle_order(
        env_state: dict[str, Any],
        action: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Place a regular order — scheduled delivery in 3-7 days."""
        product_id: str = action.get("product_id") or "electronics"
        supplier_id: str = action.get("supplier_id") or "supplier_A"
        quantity: int = action.get("quantity") or 100

        cost = quantity * UNIT_ORDER_COST
        # Supplier reliability may cause partial fulfilment
        reliability = env_state["supplier_reliability"].get(supplier_id, 0.80)
        effective_qty = max(1, int(round(quantity * reliability)))

        eta_days: int = int(rng.randint(3, 8))  # 3-7 inclusive

        env_state["pending_orders"].append(
            {
                "supplier": supplier_id,
                "product": product_id,
                "qty": effective_qty,
                "eta_days": eta_days,
            }
        )
        env_state["_period_order_costs"] = (
            env_state.get("_period_order_costs", 0.0) + cost
        )
        return env_state

    @staticmethod
    def _handle_negotiate(
        env_state: dict[str, Any],
        action: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Negotiate with a supplier — improves reliability by 0.02-0.05."""
        supplier_id: str = action.get("supplier_id") or "supplier_A"

        improvement = rng.uniform(0.02, 0.05)
        current = env_state["supplier_reliability"].get(supplier_id, 0.80)
        env_state["supplier_reliability"][supplier_id] = min(
            1.0, current + improvement,
        )
        env_state["_period_order_costs"] = (
            env_state.get("_period_order_costs", 0.0) + NEGOTIATION_FEE
        )
        return env_state

    @staticmethod
    def _handle_reroute(
        env_state: dict[str, Any],
        action: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Redirect the oldest pending order to a different supplier."""
        new_supplier: str = action.get("supplier_id") or "supplier_B"

        pending = env_state["pending_orders"]
        if pending:
            # Reroute the oldest order
            pending[0]["supplier"] = new_supplier
            # Re-roll ETA for the new supplier
            pending[0]["eta_days"] = int(rng.randint(3, 8))

        env_state["_period_order_costs"] = (
            env_state.get("_period_order_costs", 0.0) + REROUTE_FEE
        )
        return env_state

    @staticmethod
    def _handle_hold(env_state: dict[str, Any]) -> dict[str, Any]:
        """Do nothing — apply a small penalty."""
        env_state["_period_order_costs"] = (
            env_state.get("_period_order_costs", 0.0) + abs(HOLD_PENALTY)
        )
        return env_state

    @staticmethod
    def _handle_emergency_source(
        env_state: dict[str, Any],
        action: dict[str, Any],
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        """Instant delivery at 3× normal cost."""
        product_id: str = action.get("product_id") or "electronics"
        quantity: int = action.get("quantity") or 100

        cost = quantity * UNIT_ORDER_COST * EMERGENCY_COST_MULTIPLIER

        # Instant delivery — add directly to inventory
        env_state["inventory_levels"][product_id] = (
            env_state["inventory_levels"].get(product_id, 0) + quantity
        )
        env_state["_period_order_costs"] = (
            env_state.get("_period_order_costs", 0.0) + cost
        )
        return env_state
