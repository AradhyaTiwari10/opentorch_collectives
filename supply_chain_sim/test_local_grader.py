"""Local test to debug grader scores for all 3 tasks."""
from server.app import app
from fastapi.testclient import TestClient
from server.grader import grade_task
import itertools

client = TestClient(app)

# ===== TASK 2: supplier_negotiation =====
print("=" * 60)
print("TASK 2: supplier_negotiation")
print("=" * 60)

r = client.post("/reset", json={"episode_id": "supplier_negotiation", "seed": 42})
obs = r.json()["observation"]
trajectory = [obs]

suppliers = itertools.cycle(["supplier_A", "supplier_B", "supplier_C", "supplier_D"])
for i in range(20):
    sup = next(suppliers)
    sr = client.post("/step", json={"action": {"action_type": "negotiate", "supplier_id": sup}})
    data = sr.json()
    if sr.status_code != 200:
        print(f"  Step {i} FAILED: {sr.status_code}")
        break
    obs = data.get("observation", {})
    trajectory.append(obs)
    rel = obs.get("supplier_reliability", {})
    cash = obs.get("cash_balance", 0)
    done = data.get("done", False)
    print(f"  Step {i}: mean_rel={sum(rel.values())/len(rel):.4f}, cash={cash:.0f}, done={done}")
    if done:
        break

result = grade_task("supplier_negotiation", trajectory)
print(f"\nLocal grade: {result}")

grade_resp = client.post("/grade/supplier_negotiation", json=trajectory)
print(f"HTTP grade: {grade_resp.json()}")

# ===== TASK 3: disruption_response =====
print("\n" + "=" * 60)
print("TASK 3: disruption_response")
print("=" * 60)

r = client.post("/reset", json={"episode_id": "disruption_response", "seed": 42})
obs = r.json()["observation"]
trajectory3 = [obs]

for i in range(25):
    action = {"action_type": "emergency_source", "product_id": "electronics", "quantity": 200}
    sr = client.post("/step", json={"action": action})
    data = sr.json()
    if sr.status_code != 200:
        print(f"  Step {i} FAILED: {sr.status_code}")
        break
    obs = data.get("observation", {})
    trajectory3.append(obs)
    sl = obs.get("service_level", 0)
    cash = obs.get("cash_balance", 0)
    disrupt = obs.get("disruption_active", False)
    dtype = obs.get("disruption_type", None)
    done = data.get("done", False)
    print(f"  Step {i}: service_level={sl:.2f}, cash={cash:.0f}, disruption={disrupt}/{dtype}, done={done}")
    if done:
        break

result3 = grade_task("disruption_response", trajectory3)
print(f"\nLocal grade: {result3}")

grade_resp3 = client.post("/grade/disruption_response", json=trajectory3)
print(f"HTTP grade: {grade_resp3.json()}")
