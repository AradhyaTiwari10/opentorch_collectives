"""Direct test of grader+environment without HTTP layer."""
import sys
sys.path.insert(0, ".")

import numpy as np
from server.environment import SupplyChainEnvironment
from server.models import SupplyChainAction
from server.grader import grade_task

def obs_to_dict(obs):
    return obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

# ─── TASK 2: supplier_negotiation ─────────────────────────────────────
print("=" * 60)
print("TASK 2: supplier_negotiation")
print("=" * 60)

env = SupplyChainEnvironment()
obs = env.reset(seed=42, episode_id="supplier_negotiation")
trajectory = [obs_to_dict(obs)]

import itertools
suppliers = itertools.cycle(["supplier_A","supplier_B","supplier_C","supplier_D"])
for i in range(20):
    sup = next(suppliers)
    action = SupplyChainAction(action_type="negotiate", supplier_id=sup)
    obs = env.step(action)
    d = obs_to_dict(obs)
    trajectory.append(d)
    rel = d.get("supplier_reliability", {})
    mean_rel = sum(rel.values())/len(rel) if rel else 0
    cash = d.get("cash_balance", 0)
    done = d.get("done", False)
    print(f"  Step {i}: mean_rel={mean_rel:.4f}, cash={cash:.0f}, done={done}")
    if done:
        break

result = grade_task("supplier_negotiation", trajectory)
print(f"\nGrade: score={result.score:.4f}, passed={result.passed}")
print(f"Details: {result.details}")

# ─── TASK 3: disruption_response ──────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: disruption_response")
print("=" * 60)

env3 = SupplyChainEnvironment()
obs3 = env3.reset(seed=42, episode_id="disruption_response")
trajectory3 = [obs_to_dict(obs3)]

for i in range(25):
    # Mix orders + emergency sourcing
    if i % 2 == 0:
        action = SupplyChainAction(action_type="order", product_id="electronics",
                                   supplier_id="supplier_A", quantity=300)
    else:
        action = SupplyChainAction(action_type="emergency_source", product_id="food",
                                   quantity=200)
    obs3 = env3.step(action)
    d = obs_to_dict(obs3)
    trajectory3.append(d)
    sl = d.get("service_level", 0)
    cash = d.get("cash_balance", 0)
    disrupt = d.get("disruption_active", False)
    dtype = d.get("disruption_type", None)
    done = d.get("done", False)
    print(f"  Step {i}: sl={sl:.2f}, cash={cash:.0f}, disrupt={disrupt}/{dtype}, done={done}")
    if done:
        break

result3 = grade_task("disruption_response", trajectory3)
print(f"\nGrade: score={result3.score:.4f}, passed={result3.passed}")
print(f"Details: {result3.details}")
