"""Inference script for the Supply Chain Sim agent.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    SPACE_URL      The URL of the deployed HF Space environment.

STDOUT FORMAT — parsed by the automated evaluation pipeline:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys
import urllib.request
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Reads the local .env file (harmless in CI)

# ── Constants ──────────────────────────────────────────────────────────────

ENV_NAME = "supply-chain-sim"
SUCCESS_SCORE_THRESHOLD = 0.5   # score >= this → success=true in [END]
MAX_STEPS = 30                  # hard ceiling per task to stay under 20 min


# ── Mandatory structured log helpers ──────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit one [START] line at the beginning of each episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit one [STEP] line immediately after env.step() returns."""
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    # action must be a single-line string — strip any newlines
    action_str = str(action).replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """Emit one [END] line after env.close() / grading — always emitted."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────

def post_json(url: str, data: dict | list = None) -> dict:
    """Send a JSON POST request and return a decoded JSON response."""
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


# ── Action extraction ──────────────────────────────────────────────────────

def extract_action(response_text: str) -> dict:
    """Parse the LLM response into a SupplyChainAction-compatible dict.

    Falls back to {"action_type": "hold"} if parsing fails.
    """
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            action_json = response_text[start_idx : end_idx + 1]
            action = json.loads(action_json)
            valid_types = {"order", "negotiate", "reroute", "hold", "emergency_source"}
            if action.get("action_type") not in valid_types:
                return {"action_type": "hold"}
            return action
    except Exception:
        pass
    return {"action_type": "hold"}


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(task_id: str, obs: dict, step: int) -> tuple[str, str]:
    """Return (system_prompt, user_content) for the LLM call."""
    inv = obs.get("inventory_levels", {})
    rel = obs.get("supplier_reliability", {})
    period = obs.get("period", 0)
    disruption = obs.get("disruption_active", False)
    dtype = obs.get("disruption_type", None)

    low_prod = min(inv, key=inv.get) if inv else "electronics"
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
            example = (
                '{"action_type": "emergency_source", "product_id": "' + low_prod + '", "quantity": 400}'
            )
        elif period < 3:
            strategy = (
                "PREPARATION: A port_strike disruption hits at period 3. "
                "Pre-stock ALL products now by ordering large quantities. "
                f"Order '{low_prod}' from 'supplier_A' — build buffer stock NOW."
            )
            example = (
                '{"action_type": "order", "product_id": "' + low_prod + '", '
                '"supplier_id": "supplier_A", "quantity": 600}'
            )
        elif period < 8:
            strategy = (
                "PREPARATION: A supplier_failure disruption hits at period 8. "
                "Pre-stock by ordering from multiple suppliers. "
                f"Order '{low_prod}' from 'supplier_B' to diversify supply."
            )
            example = (
                '{"action_type": "order", "product_id": "' + low_prod + '", '
                '"supplier_id": "supplier_B", "quantity": 500}'
            )
        else:
            strategy = (
                "Post-disruption phase. Keep restocking to maintain service levels. "
                f"Order '{low_prod}' from the best available supplier."
            )
            example = (
                '{"action_type": "order", "product_id": "' + low_prod + '", '
                '"supplier_id": "supplier_A", "quantity": 400}'
            )

    system_prompt = (
        "You are an expert AI supply chain manager.\n"
        f"STRATEGY: {strategy}\n\n"
        "OUTPUT FORMAT: Respond with ONLY a single valid JSON object. "
        "Valid action_type values: order, negotiate, hold, emergency_source, reroute.\n"
        "For 'order': include product_id, supplier_id, quantity.\n"
        "For 'negotiate': include supplier_id.\n"
        "For 'emergency_source': include product_id, quantity.\n"
        'For \'hold\': just {"action_type": "hold"}.\n\n'
        f"EXAMPLE OUTPUT:\n{example}"
    )

    return system_prompt, json.dumps(obs)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Defaults required by hackathon spec — reflect active inference setup
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    HF_TOKEN     = os.getenv("HF_TOKEN")
    SPACE_URL    = os.getenv("SPACE_URL", "https://aradhya10-supply-chain-sim.hf.space")

    SPACE_URL = SPACE_URL.rstrip("/")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["inventory_management", "supplier_negotiation", "disruption_response"]
    all_scores: list[float] = []

    for task_id in tasks:
        # ── Emit [START] ────────────────────────────────────────────────
        log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

        step_rewards: list[float] = []
        step = 0
        score = 0.0
        last_error: Optional[str] = None

        # 1. Reset environment
        reset_url = f"{SPACE_URL}/reset"
        try:
            reset_resp = post_json(reset_url, {"episode_id": task_id, "seed": 42})
            if "observation" not in reset_resp:
                log_end(success=False, steps=0, score=0.0, rewards=[])
                all_scores.append(0.0)
                continue
            obs = reset_resp["observation"]
        except Exception as exc:
            last_error = str(exc)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            all_scores.append(0.0)
            continue

        trajectory = [obs]
        done = False

        # 2. Agent loop
        while not done and step < MAX_STEPS:
            system_prompt, user_content = build_prompt(task_id, obs, step)
            last_error = None

            # Ask the LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=100,
                    temperature=0.1,
                )
                response_text = response.choices[0].message.content
                action = extract_action(response_text)
            except Exception as exc:
                last_error = str(exc)
                action = {"action_type": "hold"}

            # Step the environment
            step_url = f"{SPACE_URL}/step"
            step_reward = 0.0
            try:
                step_resp = post_json(step_url, {"action": action})
                if "observation" not in step_resp:
                    last_error = f"no observation in step response: {step_resp}"
                    done = True
                else:
                    obs = step_resp["observation"]
                    trajectory.append(obs)
                    done = bool(step_resp.get("done", False))
                    # Capture the per-step reward from the observation
                    step_reward = float(obs.get("reward", 0.0))
            except Exception as exc:
                last_error = str(exc)
                done = True

            step_rewards.append(step_reward)
            step += 1

            # ── Emit [STEP] ──────────────────────────────────────────────
            log_step(
                step=step,
                action=json.dumps(action),
                reward=step_reward,
                done=done,
                error=last_error,
            )

        # 3. Grade the episode
        grade_url = f"{SPACE_URL}/grade/{task_id}"
        try:
            grade_resp = post_json(grade_url, trajectory)
            score = float(grade_resp.get("score", 0.0))
        except Exception as exc:
            score = 0.0
            last_error = str(exc)

        all_scores.append(score)
        success = score >= SUCCESS_SCORE_THRESHOLD

        # ── Emit [END] ───────────────────────────────────────────────────
        log_end(
            success=success,
            steps=step,
            score=score,
            rewards=step_rewards,
        )

    # 4. Final summary (human-readable, not parsed by automated eval)
    score_1 = all_scores[0] if len(all_scores) > 0 else 0.0
    score_2 = all_scores[1] if len(all_scores) > 1 else 0.0
    score_3 = all_scores[2] if len(all_scores) > 2 else 0.0
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\n{'='*50}", flush=True)
    print(f"Task 1 - inventory_management : {score_1:.3f}", flush=True)
    print(f"Task 2 - supplier_negotiation : {score_2:.3f}", flush=True)
    print(f"Task 3 - disruption_response  : {score_3:.3f}", flush=True)
    print(f"Overall Mean Score            : {mean_score:.3f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
