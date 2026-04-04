# SupplyChainSim — OpenEnv Environment

## Environment Description
The SupplyChainSim modeling environment places an AI agent in the role of a global supply chain operations manager. The agent is responsible for optimizing procurement, inventory, logistics, and supplier decisions under significant uncertainty. Operating a multi-product supply chain involving various items (electronics, apparel, food, medical, automotive) and multiple suppliers, the agent must efficiently balance stochastic demand and varying supplier reliability.

This simulation is highly valuable for Reinforcement Learning (RL) research as it tests multi-objective decision-making in complex, dynamic scenarios. The system requires balancing competing constraints, such as maximizing the fulfillment service level while minimizing holding and stockout costs. Importantly, it evaluates long-term resilience by introducing out-of-distribution events like port strikes and supplier failures, creating a robust testbed for real-world logistical planning and anomaly recovery strategies.

## Observation Space
| field name | type | description |
| :--- | :--- | :--- |
| `inventory_levels` | `dict[str, int]` | product_id → units currently in stock |
| `pending_orders` | `list[dict]` | List of pending orders (supplier, product, qty, eta_days) |
| `supplier_reliability` | `dict[str, float]` | supplier_id → reliability score (0.0–1.0) |
| `demand_forecast` | `dict[str, int]` | product_id → forecasted demand for the next period |
| `current_costs` | `float` | Total cost incurred this period |
| `service_level` | `float` | Percentage of demand fulfilled (0.0–1.0) |
| `period` | `int` | Current time step in the simulation |
| `disruption_active` | `bool` | Whether a disruption event is currently happening |
| `disruption_type` | `str` | Type of active disruption (e.g., port_strike, supplier_failure) |
| `cash_balance` | `float` | Available cash balance |
| `message` | `str` | Human-readable description of the current state |

## Action Space  
| action_type | parameters | effect |
| :--- | :--- | :--- |
| `order` | `product_id`, `supplier_id`, `quantity` | Places an order with 3-7 days ETA subject to supplier reliability. |
| `negotiate` | `supplier_id` | Pays $1,000 fee to increase supplier reliability by 0.02-0.05. |
| `reroute` | `supplier_id` | Pays $2,000 fee to redirect the oldest pending order to a new supplier with rerolled ETA. |
| `hold` | none | Do nothing. Incurs a small $5 structural penalty. |
| `emergency_source` | `product_id`, `quantity` | Instant delivery ensuring inventory but at 2× the normal order cost. |

## Reward Function
The total reward is a weighted sum of normalized cost, service efficiency, and disruption resilience metrics:
`total_reward = (0.4 * cost_score) + (0.4 * service_score) + (0.2 * resilience_score)`

**Explanation of Weights:**
- **cost_score**: Formula `max(0.0, min(1.0, 1.0 - (period_cost / 50000.0)))`. Incentivizes keeping period costs below baseline. Weight (0.4) pushes the agent to optimize structural spending over time.
- **service_score**: Formula `max(0.0, min(1.0, service_level))`. Maximizes demand fulfilled successfully. Equal weight (0.4) creates a tension with cost—stockouts decrease service score but ordering too much increases holding costs.
- **resilience_score**: Reflects adaptation during active disruptions. Base is `1.0`. Drops during an event, but actions like `emergency_source` or `reroute` limit the drop to `0.8`, standard actions give `0.5`, and passive `hold` yields `0.2`, encouraging active crisis management (0.2 weight).

## Tasks
| Task ID | Difficulty | Goal | Success Criterion | Max Steps |
| :--- | :--- | :--- | :--- | :--- |
| `inventory_management` | easy | Maintain service_level >= 0.80 for 10 consecutive periods. | `service_level >= 0.8` for 10 periods | 15 |
| `supplier_negotiation` | medium | Improve average supplier reliability from 0.86 to >= 0.92 while keeping cash_balance > 200,000. | `reliability >= 0.92` & `cash > 200000` | 20 |
| `disruption_response` | hard | Recover service_level to >= 0.75 within 5 periods of each forced disruption (port_strike at period 3, supplier_failure at period 8). | `service_level >= 0.75` within 5 periods of both events | 25 |

## Baseline Scores
Scores produced by running `inference.py` with `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Providers against the live HF Space (seed=42).

| Task | Baseline Score |
| :--- | :--- |
| `inventory_management` | 0.500 |
| `supplier_negotiation` | 0.416 |
| `disruption_response` | 0.700 |
| **Overall Mean** | **0.539** |

## Setup & Usage

**Docker Environment**
```bash
docker build -t supply-chain-sim .
docker run -p 7860:7860 supply-chain-sim
```

**Run Agent script**
```bash
export API_BASE_URL="your-llm-base-url"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export SPACE_URL="http://localhost:7860"

python inference.py
```

**Health & Endpoints Test**
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
curl http://localhost:7860/tasks
```

## OpenEnv Compliance
- openenv validate: ✅ passes
- Dockerfile: ✅ builds  
- HF Space: ✅ deployed at https://aradhya10-supply-chain-sim.hf.space
