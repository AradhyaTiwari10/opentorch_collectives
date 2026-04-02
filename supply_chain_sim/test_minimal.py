"""Minimal test of grader logic without importing openenv."""
import sys, os
sys.path.insert(0, ".")

# Patch out the openenv dependency so we can test dynamics+grader directly
import types

# Create minimal stubs
env_mod = types.ModuleType("openenv")
core_mod = types.ModuleType("openenv.core")
es_mod = types.ModuleType("openenv.core.env_server")
types_mod = types.ModuleType("openenv.core.env_server.types")

class FakeBase:
    pass

class FakeObservation(FakeBase):
    done: bool = False
    reward: float = 0.0

class FakeAction(FakeBase):
    pass

types_mod.Action = FakeAction
types_mod.Observation = FakeObservation
es_mod.Environment = FakeBase
es_mod.types = types_mod

sys.modules["openenv"] = env_mod
sys.modules["openenv.core"] = core_mod
sys.modules["openenv.core.env_server"] = es_mod
sys.modules["openenv.core.env_server.types"] = types_mod

# Now import our modules
from server.dynamics import SupplyChainDynamics, BASE_DEMAND
from server.tasks import TASK_REGISTRY
from server.grader import grade_task
import numpy as np

def make_state(task_id, seed=42):
    task = TASK_REGISTRY[task_id]
    products = ["electronics","apparel","food","medical","automotive"]
    reliabilities = {"supplier_A":0.95,"supplier_B":0.85,"supplier_C":0.75,"supplier_D":0.90}
    return {
        "inventory_levels": {p: 1000 for p in products},
        "pending_orders": [],
        "supplier_reliability": dict(reliabilities),
        "demand_forecast": {p: BASE_DEMAND[p] for p in products},
        "current_costs": 0.0,
        "service_level": 1.0,
        "period": 0,
        "disruption_active": False,
        "disruption_type": None,
        "cash_balance": task.initial_cash,
        "max_steps": task.max_steps,
        "_period_order_costs": 0.0,
    }

dyn = SupplyChainDynamics()

# ─── TASK 2 ────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 2: supplier_negotiation (negotiate all suppliers)")
print("=" * 60)

rng = np.random.RandomState(seed=42)
state = make_state("supplier_negotiation")
trajectory = [dict(state)]

import itertools
suppliers = itertools.cycle(["supplier_A","supplier_B","supplier_C","supplier_D"])
for i in range(20):
    sup = next(suppliers)
    state = dyn.apply_action(state, {"action_type":"negotiate","supplier_id":sup}, rng)
    state = dyn.advance_period(state, rng)
    trajectory.append(dict(state))
    rel = state["supplier_reliability"]
    mean_rel = sum(rel.values())/len(rel)
    print(f"  Step {i}: mean_rel={mean_rel:.4f}, cash={state['cash_balance']:.0f}, done={dyn.check_done(state)}")
    if dyn.check_done(state):
        break

result = grade_task("supplier_negotiation", trajectory)
print(f"\nGrade: score={result.score:.4f}, details={result.details}")

# ─── TASK 3 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: disruption_response (mixed order + emergency)")
print("=" * 60)

rng3 = np.random.RandomState(seed=42)
state3 = make_state("disruption_response")
trajectory3 = [dict(state3)]

products = ["electronics","apparel","food","medical","automotive"]
for i in range(25):
    # Smart strategy: order all products each turn
    if i % 3 == 0:
        action = {"action_type":"order","product_id":"electronics","supplier_id":"supplier_A","quantity":300}
    elif i % 3 == 1:
        action = {"action_type":"order","product_id":"food","supplier_id":"supplier_B","quantity":300}
    else:
        action = {"action_type":"order","product_id":"apparel","supplier_id":"supplier_A","quantity":200}
    state3 = dyn.apply_action(state3, action, rng3)
    state3 = dyn.advance_period(state3, rng3)
    trajectory3.append(dict(state3))
    print(f"  Step {i}: sl={state3['service_level']:.2f}, cash={state3['cash_balance']:.0f}, period={state3['period']}, done={dyn.check_done(state3)}")
    if dyn.check_done(state3):
        break

result3 = grade_task("disruption_response", trajectory3)
print(f"\nGrade: score={result3.score:.4f}, details={result3.details}")
