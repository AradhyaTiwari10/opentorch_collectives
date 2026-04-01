"""Inference script for the Supply Chain Sim agent.

Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env vars.
Connects to the live HF Space URL via SPACE_URL env var (never localhost).
"""

import os
import json
import urllib.request
from openai import OpenAI

def post_json(url: str, data: dict | list = None) -> dict:
    """Helper to send a JSON POST request and receive a JSON response."""
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    
    if data is not None:
        raw_data = json.dumps(data).encode("utf-8")
        req.add_header("Content-Length", str(len(raw_data)))
        with urllib.request.urlopen(req, data=raw_data) as response:
            return json.loads(response.read().decode("utf-8"))
    else:
        req.add_header("Content-Length", "0")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

def extract_action(response_text: str) -> dict:
    """Parse the LLM response to extract the JSON action matching SupplyChainAction schema."""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            action_json = response_text[start_idx:end_idx+1]
            action = json.loads(action_json)
            valid_types = {"order", "negotiate", "reroute", "hold", "emergency_source"}
            if action.get("action_type") not in valid_types:
                return {"action_type": "hold"}
            return action
    except Exception:
        pass
    return {"action_type": "hold"}

def build_prompt(task_id: str, obs: dict, step: int) -> tuple[str, str]:
    """Return (system_prompt, user_content) for the LLM call."""
    products = ["electronics", "apparel", "food", "medical", "automotive"]
    suppliers = ["supplier_A", "supplier_B", "supplier_C", "supplier_D"]
    
    # Figure out which product/supplier needs most attention
    inv = obs.get("inventory_levels", {})
    rel = obs.get("supplier_reliability", {})
    period = obs.get("period", 0)
    disruption = obs.get("disruption_active", False)
    dtype = obs.get("disruption_type", None)
    
    # Lowest-stock product
    low_prod = min(inv, key=inv.get) if inv else "electronics"
    # Lowest-reliability supplier  
    low_sup = min(rel, key=rel.get) if rel else "supplier_C"

    if task_id == "inventory_management":
        strategy = (
            "GOAL: Keep service_level >= 0.80 every period. "
            "RULE: If any product has inventory < 500, ORDER it immediately. "
            f"Current lowest stock product: '{low_prod}' with {inv.get(low_prod, 0)} units. "
            "Order 400-600 units of the lowest-stock product each turn using the most reliable supplier."
        )
        example = (
            '{"action_type": "order", "product_id": "' + low_prod + '", '
            '"supplier_id": "supplier_A", "quantity": 500}'
        )

    elif task_id == "supplier_negotiation":
        # Alternate negotiate with hold to conserve cash
        # Negotiate on steps 0,2,4... hold on 1,3,5...
        if step % 2 == 0:
            strategy = (
                "GOAL: Raise average supplier reliability to >= 0.92. "
                f"The lowest reliability supplier is '{low_sup}' at {rel.get(low_sup, 0):.3f}. "
                "Use negotiate action on the lowest-reliability supplier to improve it."
            )
            example = '{"action_type": "negotiate", "supplier_id": "' + low_sup + '"}'
        else:
            strategy = (
                "GOAL: Conserve cash while maintaining suppliers. "
                "Use hold this turn to save money. Reserve negotiate for next turn."
            )
            example = '{"action_type": "hold"}'

    else:  # disruption_response
        if disruption:
            strategy = (
                f"CRITICAL DISRUPTION: {dtype} is active! "
                f"Use emergency_source on '{low_prod}' to immediately restock and recover service level. "
                "This is urgent — emergency sourcing gives instant inventory."
            )
            example = ('{"action_type": "emergency_source", "product_id": "' + low_prod +
                       '", "quantity": 400}')
        elif period < 3:
            # Pre-stock before port_strike at period 3
            strategy = (
                "PREPARATION: A port_strike disruption hits at period 3. "
                "Pre-stock ALL products now by ordering large quantities. "
                f"Order '{low_prod}' from 'supplier_A' — build buffer stock NOW."
            )
            example = ('{"action_type": "order", "product_id": "' + low_prod +
                       '", "supplier_id": "supplier_A", "quantity": 600}')
        elif period < 8:
            # Pre-stock before supplier_failure at period 8
            strategy = (
                "PREPARATION: A supplier_failure disruption hits at period 8. "
                "Pre-stock by ordering from multiple suppliers. "
                f"Order '{low_prod}' from 'supplier_B' to diversify supply."
            )
            example = ('{"action_type": "order", "product_id": "' + low_prod +
                       '", "supplier_id": "supplier_B", "quantity": 500}')
        else:
            strategy = (
                "Post-disruption phase. Keep restocking to maintain service levels. "
                f"Order '{low_prod}' from the best available supplier."
            )
            example = ('{"action_type": "order", "product_id": "' + low_prod +
                       '", "supplier_id": "supplier_A", "quantity": 400}')

    system_prompt = (
        "You are an expert AI supply chain manager.\n"
        f"STRATEGY: {strategy}\n\n"
        "OUTPUT FORMAT: Respond with ONLY a single valid JSON object. "
        "Valid action_type values: order, negotiate, hold, emergency_source, reroute.\n"
        "For 'order': include product_id, supplier_id, quantity.\n"
        "For 'negotiate': include supplier_id.\n"
        "For 'emergency_source': include product_id, quantity.\n"
        "For 'hold': just {\"action_type\": \"hold\"}.\n\n"
        f"EXAMPLE OUTPUT:\n{example}"
    )
    
    return system_prompt, json.dumps(obs)

def main() -> None:
    API_BASE_URL = os.getenv("API_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    SPACE_URL = os.getenv("SPACE_URL", "https://YOUR_USERNAME-supply-chain-sim.hf.space")

    if not all([API_BASE_URL, MODEL_NAME, SPACE_URL]):
        print("Missing required environment variables. Please set API_BASE_URL, MODEL_NAME, and SPACE_URL.")
        return

    SPACE_URL = SPACE_URL.rstrip('/')
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["inventory_management", "supplier_negotiation", "disruption_response"]
    scores = []

    for task_id in tasks:
        print(f"\n{'='*50}")
        print(f"Running task: {task_id}")
        print(f"{'='*50}")
        
        # 1. Reset env
        reset_url = f"{SPACE_URL}/reset"
        try:
            reset_resp = post_json(reset_url, {"episode_id": task_id, "seed": 42})
            if "observation" not in reset_resp:
                print(f"Reset failed: {reset_resp}")
                scores.append(0.0)
                continue
            obs = reset_resp["observation"]
        except Exception as e:
            print(f"Reset error: {e}")
            scores.append(0.0)
            continue
            
        trajectory = [obs]
        step = 0

        # 2. Run agent loop
        while True:
            system_prompt, user_content = build_prompt(task_id, obs, step)
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                )
                response_text = response.choices[0].message.content
                action = extract_action(response_text)
                print(f"  Step {step}: {action}")
            except Exception as e:
                print(f"  Step {step} LLM ERROR: {e}")
                action = {"action_type": "hold"}
            
            # Step env
            step_url = f"{SPACE_URL}/step"
            try:
                step_resp = post_json(step_url, {"action": action})
                if "observation" not in step_resp:
                    print(f"  Step {step} NO OBS: {step_resp}")
                    break
                obs = step_resp["observation"]
            except Exception as e:
                print(f"  Step {step} STEP ERROR: {e}")
                break
                
            trajectory.append(obs)
            step += 1
            
            # Print key metrics
            sl = obs.get("service_level", 0)
            cash = obs.get("cash_balance", 0)
            rel = obs.get("supplier_reliability", {})
            mean_rel = sum(rel.values())/len(rel) if rel else 0
            print(f"    -> sl={sl:.2f}, cash={cash:.0f}, mean_rel={mean_rel:.3f}")
            
            if step_resp.get("done", False):
                print(f"  Episode done at step {step}")
                break
        
        # 3. Grade
        grade_url = f"{SPACE_URL}/grade/{task_id}"
        try:
            grade_resp = post_json(grade_url, trajectory)
            score = grade_resp.get("score", 0.0)
            print(f"  Grade details: {grade_resp}")
        except Exception as e:
            print(f"  Grade error: {e}")
            score = 0.0
            
        scores.append(score)

    # 4. Print final scores
    score_1 = scores[0] if len(scores) > 0 else 0.0
    score_2 = scores[1] if len(scores) > 1 else 0.0
    score_3 = scores[2] if len(scores) > 2 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*50}")
    print(f"Task 1 - inventory_management: {score_1:.3f}")
    print(f"Task 2 - supplier_negotiation: {score_2:.3f}")
    print(f"Task 3 - disruption_response: {score_3:.3f}")
    print(f"Overall: {mean_score:.3f}")

if __name__ == "__main__":
    main()
