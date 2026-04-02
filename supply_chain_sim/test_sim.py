"""Simulate expected scores with new grader formulas."""
import sys
sys.path.insert(0, ".")

import types

# Stub openenv
env_mod = types.ModuleType("openenv")
core_mod = types.ModuleType("openenv.core")
es_mod = types.ModuleType("openenv.core.env_server")
types_mod = types.ModuleType("openenv.core.env_server.types")

class FakeBase: pass
class FakeObs(FakeBase):
    done = False
    reward = 0.0
class FakeAct(FakeBase): pass

types_mod.Action = FakeAct
types_mod.Observation = FakeObs
es_mod.Environment = FakeBase
es_mod.types = types_mod

sys.modules["openenv"] = env_mod
sys.modules["openenv.core"] = core_mod
sys.modules["openenv.core.env_server"] = es_mod
sys.modules["openenv.core.env_server.types"] = types_mod

from server.dynamics import SupplyChainDynamics, BASE_DEMAND
from server.tasks import TASK_REGISTRY
from server.grader import grade_task
import numpy as np
import itertools

def make_state(task_id, seed=42):
    task = TASK_REGISTRY[task_id]
    products = ["electronics","apparel","food","medical","automotive"]
    return {
        "inventory_levels": {p: 1000 for p in products},
        "pending_orders": [],
        "supplier_reliability": {"supplier_A":0.95,"supplier_B":0.85,"supplier_C":0.75,"supplier_D":0.90},
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

print("=" * 60)
print("TASK 1: inventory_management (order lowest-stock product)")
print("=" * 60)

rng = np.random.RandomState(42)
state = make_state("inventory_management")
trajectory = [dict(state)]
products = ["electronics","apparel","food","medical","automotive"]

for i in range(15):
    inv = state["inventory_levels"]
    low_prod = min(inv, key=inv.get)
    action = {"action_type":"order","product_id":low_prod,"supplier_id":"supplier_A","quantity":500}
    state = dyn.apply_action(state, action, rng)
    state = dyn.advance_period(state, rng)
    trajectory.append(dict(state))
    print(f"  Step {i}: sl={state['service_level']:.2f}, cash={state['cash_balance']:.0f}, done={dyn.check_done(state)}")
    if dyn.check_done(state): break

r1 = grade_task("inventory_management", trajectory)
print(f"Grade: score={r1.score:.4f}, details={r1.details}")

print("\n" + "=" * 60)
print("TASK 2: supplier_negotiation (alternate negotiate/hold)")
print("=" * 60)

rng2 = np.random.RandomState(42)
state2 = make_state("supplier_negotiation")
trajectory2 = [dict(state2)]

for i in range(20):
    rel = state2["supplier_reliability"]
    low_sup = min(rel, key=rel.get)
    if i % 2 == 0:
        action = {"action_type":"negotiate","supplier_id":low_sup}
    else:
        action = {"action_type":"hold"}
    state2 = dyn.apply_action(state2, action, rng2)
    state2 = dyn.advance_period(state2, rng2)
    trajectory2.append(dict(state2))
    mean_rel = sum(rel.values())/len(rel)
    print(f"  Step {i}: mean_rel={mean_rel:.4f}, cash={state2['cash_balance']:.0f}, done={dyn.check_done(state2)}")
    if dyn.check_done(state2): break

r2 = grade_task("supplier_negotiation", trajectory2)
print(f"Grade: score={r2.score:.4f}, details={r2.details}")

print("\n" + "=" * 60)
print("TASK 3: disruption_response (pre-stock + emergency)")
print("=" * 60)

rng3 = np.random.RandomState(42)
state3 = make_state("disruption_response")
trajectory3 = [dict(state3)]

prod_cycle = itertools.cycle(products)
for i in range(25):
    period = state3["period"]
    inv = state3["inventory_levels"]
    low_prod = min(inv, key=inv.get)
    disrupt = state3["disruption_active"]
    
    if disrupt:
        # Emergency source lowest-stock product
        action = {"action_type":"emergency_source","product_id":low_prod,"quantity":400}
    elif period < 3:
        # Pre-stock before port_strike at period 3
        action = {"action_type":"order","product_id":low_prod,"supplier_id":"supplier_A","quantity":600}
    elif period < 8:
        # Continue stocking before supplier_failure at period 8
        prod = next(prod_cycle)
        action = {"action_type":"order","product_id":prod,"supplier_id":"supplier_B","quantity":500}
    else:
        action = {"action_type":"order","product_id":low_prod,"supplier_id":"supplier_A","quantity":400}
    
    state3 = dyn.apply_action(state3, action, rng3)
    state3 = dyn.advance_period(state3, rng3)
    trajectory3.append(dict(state3))
    print(f"  Step {i}: period={state3['period']}, sl={state3['service_level']:.2f}, cash={state3['cash_balance']:.0f}, disrupt={disrupt}, done={dyn.check_done(state3)}")
    if dyn.check_done(state3): break

r3 = grade_task("disruption_response", trajectory3)
print(f"Grade: score={r3.score:.4f}, details={r3.details}")

print("\n" + "=" * 60)
scores = [r1.score, r2.score, r3.score]
print(f"FINAL SCORES:")
print(f"  inventory_management:  {r1.score:.3f}")
print(f"  supplier_negotiation:  {r2.score:.3f}")
print(f"  disruption_response:   {r3.score:.3f}")
print(f"  Overall mean:          {sum(scores)/len(scores):.3f}")
