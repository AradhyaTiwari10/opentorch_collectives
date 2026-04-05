"""Pydantic models for the supply chain simulation.

Note: Never declare 'done' or 'reward' fields in Observation models —
they are inherited from the base class.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation

# ---------------------------------------------------------------------------
# Reward base – openenv-core may or may not expose a Reward base class.
# We try to import it; if unavailable we fall back to plain BaseModel.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Reward as _RewardBase
except ImportError:
    _RewardBase = BaseModel  # type: ignore[assignment,misc]


# ── Observation ────────────────────────────────────────────────────────────

class SupplyChainObservation(Observation):
    """State visible to the agent each time-step.

    `done` and `reward` are inherited from the openenv Observation base
    and must NOT be redeclared here.
    """

    inventory_levels: dict[str, int] = Field(
        default_factory=dict,
        description="product_id → units currently in stock",
    )
    pending_orders: list[dict] = Field(
        default_factory=list,
        description=(
            "List of pending orders, each a dict with keys: "
            "supplier, product, qty, eta_days"
        ),
    )
    supplier_reliability: dict[str, float] = Field(
        default_factory=dict,
        description="supplier_id → reliability score (0.0–1.0)",
    )
    demand_forecast: dict[str, int] = Field(
        default_factory=dict,
        description="product_id → forecasted demand for the next period",
    )
    current_costs: float = Field(
        default=0.0,
        description="Total cost incurred this period",
    )
    service_level: float = Field(
        default=1.0,
        description="Percentage of demand fulfilled (0.0–1.0)",
    )
    period: int = Field(
        default=0,
        description="Current time step in the simulation",
    )
    disruption_active: bool = Field(
        default=False,
        description="Whether a disruption event is currently happening",
    )
    disruption_type: str | None = Field(
        default=None,
        description=(
            'Type of active disruption: "port_strike", '
            '"supplier_failure", "demand_surge", or None'
        ),
    )
    cash_balance: float = Field(
        default=0.0,
        description="Available cash balance",
    )
    carbon_footprint: float = Field(
        default=0.0,
        description=(
            "Cumulative CO2-equivalent emissions (kg) from sourcing decisions this episode. "
            "emergency_source (air freight) = 5 kg CO2e/unit; "
            "order (sea/road freight) = 1 kg CO2e/unit. "
            "Lower is better — penalised in the sustainability_score reward component."
        ),
    )
    message: str = Field(
        default="",
        description="Human-readable description of the current state",
    )


# ── Action ─────────────────────────────────────────────────────────────────

class SupplyChainAction(Action):
    """An action the agent can take each time-step."""

    action_type: str = Field(
        ...,
        description=(
            'One of: "order", "negotiate", "reroute", "hold", '
            '"emergency_source"'
        ),
    )
    product_id: str | None = Field(
        default=None,
        description="Target product identifier",
    )
    supplier_id: str | None = Field(
        default=None,
        description="Target supplier identifier",
    )
    quantity: int | None = Field(
        default=None,
        description="Number of units (for order / emergency_source)",
    )
    target_price: float | None = Field(
        default=None,
        description="Desired price per unit (for negotiate actions)",
    )
    notes: str | None = Field(
        default=None,
        description="Agent reasoning (logged but not used in the sim)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action_type": "order",
                "product_id": "electronics",
                "supplier_id": "supplier_A",
                "quantity": 100
            }
        }
    }


# ── Reward ─────────────────────────────────────────────────────────────────

class SupplyChainReward(_RewardBase):
    """Multi-objective reward returned after each step."""

    cost_score: float = Field(
        default=0.0,
        description="Normalised cost-efficiency score (0-1)",
    )
    service_score: float = Field(
        default=0.0,
        description="Demand fulfilment rate (0-1)",
    )
    resilience_score: float = Field(
        default=0.0,
        description="Recovery speed / behaviour during active disruption (0-1)",
    )
    sustainability_score: float = Field(
        default=0.0,
        description=(
            "Inverse carbon footprint score (0-1). "
            "Penalises excessive use of high-emission emergency sourcing."
        ),
    )
    total_reward: float = Field(
        default=0.0,
        description=(
            "Weighted sum: 0.35*cost + 0.35*service + 0.15*resilience + 0.15*sustainability"
        ),
    )


# ── Task Result ────────────────────────────────────────────────────────────

class TaskResult(BaseModel):
    """Result from a grader task evaluation."""

    task_id: str = Field(
        ...,
        description="Unique identifier for the evaluated task",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Grader score (0.0–1.0)",
    )
    passed: bool = Field(
        ...,
        description="Whether the task was considered passed",
    )
    details: dict = Field(
        default_factory=dict,
        description="Additional evaluation details",
    )
