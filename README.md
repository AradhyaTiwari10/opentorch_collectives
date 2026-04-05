---
title: SupplyChainSim
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
tags:
  - openenv
  - supply-chain
  - rl
  - multi-objective
  - sustainability
---

# SupplyChainSim — OpenEnv Environment

## Environment Description & Motivation

Global supply chain disruptions cost the world economy over **$4 trillion in 2021 alone** (McKinsey Global Institute). Companies like Nike, Ford, and Apple have each suffered **billions in losses** from a single supplier failure or port congestion event — yet most decisions are still made by humans working from spreadsheets.

**SupplyChainSim** places an AI agent in the role of a global supply chain operations manager responsible for procuring, stocking, and distributing five product categories (electronics, apparel, food, medical, automotive) across four suppliers — under conditions of stochastic demand, supplier unreliability, random disruptions, and sustainability pressure.

This environment directly addresses a gap in RL benchmarks: real-world supply chain management requires **simultaneous multi-objective optimisation under uncertainty**, long-horizon planning, and reactive crisis management — making it an ideal stress-test for frontier language model agents.

### What makes this environment novel

- **Forced disruption injection** — port strikes and supplier failures fire at deterministic periods, forcing agents to both pre-position inventory *and* react, unlike random-only environments
- **Full event lifecycle** — disruptions have durations; supplier reliability automatically restores after `supplier_failure` expires, meaning state management must be precise
- **Carbon footprint tracking** — `emergency_source` (equivalent to air freight) emits **5× the CO2e** of a regular sea-freight order. Carbon is exposed in the observation and penalised in the reward's `sustainability_score` component, creating a genuine tension between speed-of-recovery and environmental cost. No other OpenEnv supply chain environment has this dimension.
- **Partial-credit reward shaping** — four weighted reward components give the agent a meaningful gradient at every step

---

## Observation Space

| Field | Type | Description |
| :--- | :--- | :--- |
| `inventory_levels` | `dict[str, int]` | `product_id → units currently in stock` |
| `pending_orders` | `list[dict]` | Pending orders: `supplier`, `product`, `qty`, `eta_days` |
| `supplier_reliability` | `dict[str, float]` | `supplier_id → reliability score (0.0–1.0)` |
| `demand_forecast` | `dict[str, int]` | `product_id → forecasted demand next period` |
| `current_costs` | `float` | Total cost incurred this period |
| `service_level` | `float` | Fraction of demand fulfilled (0.0–1.0) |
| `period` | `int` | Current time step in the simulation |
| `disruption_active` | `bool` | Whether a disruption is currently active |
| `disruption_type` | `str\|null` | Active disruption type: `port_strike`, `supplier_failure`, `demand_surge` |
| `cash_balance` | `float` | Available cash balance |
| `carbon_footprint` | `float` | **Novel** — cumulative CO2e emissions (kg) from sourcing decisions |
| `message` | `str` | Human-readable state description |

---

## Action Space

| `action_type` | Parameters | Effect |
| :--- | :--- | :--- |
| `order` | `product_id`, `supplier_id`, `quantity` | Places order with 3–7 day ETA; quantity scaled by supplier reliability |
| `negotiate` | `supplier_id` | Pays $1,000 fee. Increases supplier reliability by 0.02–0.05 |
| `reroute` | `supplier_id` | Pays $2,000 fee. Redirects oldest pending order to a new supplier |
| `hold` | — | Do nothing. Incurs a $5 structural penalty |
| `emergency_source` | `product_id`, `quantity` | Instant delivery at 2× normal cost. Emits 5× carbon vs. standard order |

---

## Reward Function

The total reward is a four-component weighted sum, all components normalised to [0, 1]:

```
total_reward = 0.35 * cost_score
             + 0.35 * service_score
             + 0.15 * resilience_score
             + 0.15 * sustainability_score
```

| Component | Formula | What it measures |
| :--- | :--- | :--- |
| `cost_score` | `max(0, 1 − period_cost / 50000)` | Keeps procurement costs lean |
| `service_score` | `service_level` (fraction of demand met) | Maximises demand fulfilment |
| `resilience_score` | 1.0 normal / 0.8 proactive / 0.5 reactive / 0.2 passive-during-crisis | Rewards active crisis management |
| `sustainability_score` | `max(0, 1 − carbon_footprint / 100000)` | Penalises excessive air-freight emergency sourcing |

The cost↔service tension prevents the trivial solution of "always emergency source everything" — that would maximise service short-term but destroy cost and sustainability scores.

---

## Tasks

| Task ID | Difficulty | Goal | Success Criterion | Max Steps |
| :--- | :--- | :--- | :--- | :--- |
| `inventory_management` | **Easy** | Maintain service_level ≥ 0.80 | Longest consecutive streak of periods ≥ 0.80 reaches 10 | 15 |
| `supplier_negotiation` | **Medium** | Raise average supplier reliability from 0.8625 → ≥ 0.92 while keeping cash > $200k | `reliability ≥ 0.92` **and** `cash > 200000` | 20 |
| `disruption_response` | **Hard** | Recover service_level to ≥ 0.75 within 5 periods of each forced disruption (port_strike @ period 3, supplier_failure @ period 8) | Average recovery speed ≥ 1.0 across both events | 25 |

### Grader design notes

- `inventory_management` measures the **longest consecutive streak** above threshold, not total count — an agent cannot game it by oscillating
- `supplier_negotiation` awards reliability credit **only for improvement above the initial mean** (0.8625), so a pure hold-agent always scores 0 on the reliability component
- `disruption_response` measures recovery speed per disruption independently, then averages — partial credit if one disruption is handled well

---

## Baseline Scores

Produced by running `inference.py` with `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Providers (seed=42).

| Task | Baseline Score |
| :--- | :--- |
| `inventory_management` | 0.500 |
| `supplier_negotiation` | 0.416 |
| `disruption_response` | 0.700 |
| **Overall Mean** | **0.539** |

---

## End-to-End Operations Guide

### 1. Interactive Testing via Swagger UI (Browser)

You can manually interact with the environment directly from your browser without code.

1. **Open the interface:** Go to `http://localhost:7860/docs` (or your Hugging Face Space URL + `/docs`).
2. **Start an Episode:**
   - Expand `POST /reset` and click **Try it out**.
   - Input `{ "episode_id": "inventory_management", "seed": 42 }`.
   - Click **Execute**. Keep note of the returned `observation` state.
3. **Take Actions:**
   - Expand `POST /step` and click **Try it out**.
   - Replace the default payload with a valid action, for example:
     ```json
     {
       "action": {
         "action_type": "order",
         "product_id": "electronics",
         "supplier_id": "supplier_A",
         "quantity": 200
       }
     }
     ```
   - Click **Execute**. The response will show your new observation and reward.
4. **Grade Episode:** Once `done: true` is returned from a step, call `POST /grade/{task_id}` passing the JSON list of all observations returned so far to get your final score.

### 2. Automated Evaluator (Agent Inference)

The complete end-to-end evaluation runs via the provided `inference.py` script.

**Prerequisites:** Set up your environment variables.
```bash
export API_BASE_URL="https://router.huggingface.co/v1"  # Or your chosen provider
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"   # Or your chosen model
export HF_TOKEN="hf_your_hugging_face_token"            # Model provider token
export SPACE_URL="http://localhost:7860"                # Your deployed space or localhost
```

**Run the pipeline:**
```bash
python inference.py
```
This script will sequentially:
1. Connect to the running OpenEnv Server (via HTTP).
2. Reset the environment for each of the 3 specified tasks.
3. Hook your configured LLM directly to the environment.
4. Continuously parse LLM outputs into valid JSON Actions.
5. Print the strictly-formatted `[START]`, `[STEP]`, and `[END]` stdout logs for the grader.

### 3. Local Docker Testing

If you are developing locally, run the space exactly as the grader will:
```bash
docker build -t supply-chain-sim .
docker run -p 7860:7860 supply-chain-sim
```

### 4. Running Offline Unit Tests

To verify environment dynamics without starting the server:
```bash
pytest tests/ -v -m "not server"
```

### Run Tests (offline unit tests, no server needed)

```bash
pytest tests/ -v -m "not server"
```

---

## OpenEnv Compliance

- `openenv validate` ✅
- Dockerfile builds ✅
- HF Space deployed ✅ — https://aradhya10-supply-chain-sim.hf.space
- Baseline script produces reproducible scores ✅
- 3 tasks with deterministic graders ✅
