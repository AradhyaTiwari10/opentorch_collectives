# Supply Chain Sim

AI agent acts as a global supply chain operations manager, optimizing procurement, inventory, logistics, and supplier decisions under uncertainty.

## Tasks

| Task | Description |
|------|-------------|
| `inventory_management` | Optimize inventory levels across warehouses |
| `supplier_negotiation` | Negotiate contracts and pricing with suppliers |
| `disruption_response` | Respond to supply chain disruptions effectively |

## Quick Start

```bash
pip install -r requirements.txt
```

## Project Structure

```
supply_chain_sim/
├── inference.py              ← Agent inference script
├── openenv.yaml              ← OpenEnv configuration
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .antigravity/
│   └── rules.md              ← Project rules
└── server/
    ├── __init__.py
    ├── app.py                ← FastAPI application
    ├── environment.py        ← Environment logic
    ├── models.py             ← Pydantic models
    ├── dynamics.py           ← Supply chain dynamics
    ├── reward.py             ← Reward computation
    ├── events.py             ← Random event generation
    ├── tasks.py              ← Task definitions
    └── grader.py             ← Grading logic
```

## Tags

`openenv` · `supply-chain` · `logistics` · `multi-objective` · `rl`
