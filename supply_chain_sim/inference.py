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
            if "action_type" in action:
                return action
    except Exception:
        pass
    
    # Fallback on parse failure
    return {"action_type": "hold"}

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
    max_steps = 8
    scores = []

    for task_id in tasks:
        # 1. Reset env
        reset_url = f"{SPACE_URL}/reset"
        try:
            reset_resp = post_json(reset_url, {"episode_id": task_id, "seed": 42})
            if "observation" not in reset_resp:
                scores.append(0.0)
                continue
            obs = reset_resp["observation"]
        except Exception as e:
            scores.append(0.0)
            continue
            
        trajectory = [obs]

        # 2. Run agent loop
        for _ in range(max_steps):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a supply chain manager. Analyze the state and choose the best action."},
                        {"role": "user", "content": json.dumps(obs)}
                    ]
                )
                response_text = response.choices[0].message.content
                action = extract_action(response_text)
            except Exception:
                action = {"action_type": "hold"}
            
            # Step env
            step_url = f"{SPACE_URL}/step"
            try:
                step_resp = post_json(step_url, {"action": action})
                if "observation" not in step_resp:
                    break
                obs = step_resp["observation"]
            except Exception:
                break
                
            trajectory.append(obs)
            
            if step_resp.get("done", False):
                break
        
        # 3. After episode: call /grade/{task_id} with trajectory, get score
        grade_url = f"{SPACE_URL}/grade/{task_id}"
        try:
            grade_resp = post_json(grade_url, trajectory)
            score = grade_resp.get("score", 0.0)
        except Exception:
            score = 0.0
            
        scores.append(score)

    # 4. Print final scores
    score_1 = scores[0] if len(scores) > 0 else 0.0
    score_2 = scores[1] if len(scores) > 1 else 0.0
    score_3 = scores[2] if len(scores) > 2 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Task 1 - inventory_management: {score_1:.3f}")
    print(f"Task 2 - supplier_negotiation: {score_2:.3f}")
    print(f"Task 3 - disruption_response: {score_3:.3f}")
    print(f"Overall: {mean_score:.3f}")

if __name__ == "__main__":
    main()
