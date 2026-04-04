"""Task definitions: inventory_management, supplier_negotiation, disruption_response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single evaluation task."""

    task_id: str
    difficulty: str                       # "easy", "medium", "hard"
    goal: str                             # human-readable description
    max_steps: int
    initial_cash: float
    disruptions_enabled: bool
    event_probability_override: float     # 0.0→ no random events
    success_criteria: dict[str, Any]      # machine-readable thresholds
    forced_events: dict[int, str] = field(default_factory=dict)
    # period → event_type to forcefully inject


# ── Task 1 — EASY ─────────────────────────────────────────────────────────

INVENTORY_MANAGEMENT = TaskConfig(
    task_id="inventory_management",
    difficulty="easy",
    goal=(
        "Maintain service_level >= 0.80 for 10 consecutive periods."
    ),
    max_steps=15,
    initial_cash=500_000.0,
    disruptions_enabled=False,
    event_probability_override=0.0,
    success_criteria={
        "service_level_threshold": 0.80,
        "consecutive_periods": 10,
    },
)

# ── Task 2 — MEDIUM ───────────────────────────────────────────────────────

SUPPLIER_NEGOTIATION = TaskConfig(
    task_id="supplier_negotiation",
    difficulty="medium",
    goal=(
        "Improve average supplier reliability from 0.86 to >= 0.92 "
        "while keeping cash_balance > 200,000."
    ),
    max_steps=20,
    initial_cash=300_000.0,
    disruptions_enabled=True,
    event_probability_override=1.0,        # use default probabilities
    success_criteria={
        "reliability_target": 0.92,
        "min_cash_balance": 200_000.0,
    },
)

# ── Task 3 — HARD ─────────────────────────────────────────────────────────

DISRUPTION_RESPONSE = TaskConfig(
    task_id="disruption_response",
    difficulty="hard",
    goal=(
        "Recover service_level to >= 0.75 within 5 periods of each "
        "forced disruption (port_strike at period 3, supplier_failure "
        "at period 8)."
    ),
    max_steps=25,
    initial_cash=400_000.0,
    disruptions_enabled=True,
    event_probability_override=0.0,        # only forced events
    success_criteria={
        "service_level_threshold": 0.75,
        "recovery_window": 5,
        "disruption_periods": [3, 8],
    },
    forced_events={
        3: "port_strike",
        8: "supplier_failure",
    },
)

# ── Registry ───────────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, TaskConfig] = {
    INVENTORY_MANAGEMENT.task_id: INVENTORY_MANAGEMENT,
    SUPPLIER_NEGOTIATION.task_id: SUPPLIER_NEGOTIATION,
    DISRUPTION_RESPONSE.task_id: DISRUPTION_RESPONSE,
}


def get_task(task_id: str) -> TaskConfig:
    """Return the :class:`TaskConfig` for *task_id*, or raise ``KeyError``."""
    return TASK_REGISTRY[task_id]
