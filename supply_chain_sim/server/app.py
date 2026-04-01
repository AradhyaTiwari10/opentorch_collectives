"""FastAPI application entry point.

Uses create_fastapi_app to create the HTTP server.
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

# Set ENABLE_WEB_INTERFACE=true to provide Gradio UI at /web
os.environ["ENABLE_WEB_INTERFACE"] = "true"

# openenv-core imports
from openenv.core.env_server import create_fastapi_app

from server.environment import SupplyChainEnvironment
from server.grader import grade_task, run_all_graders
from server.models import SupplyChainAction, SupplyChainObservation, TaskResult
from server.tasks import TASK_REGISTRY

# Ensure we pass the CLASS not an instance, so each WS connection gets
# a strictly isolated instance with full session lifecycle/cleanup.
app: FastAPI = create_fastapi_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
)


@app.on_event("startup")
async def startup_event() -> None:
    """Log readiness and task names on startup."""
    tasks = ", ".join(TASK_REGISTRY.keys())
    # The requirement specifically asks for this exact string format
    print(f"SupplyChainSim v1.0 ready. Tasks: {tasks}")
    logging.info(f"SupplyChainSim v1.0 ready. Tasks: {tasks}")


# ── Custom Endpoints ────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks() -> List[Dict[str, Any]]:
    """Return all task definitions with descriptions and difficulty."""
    tasks_info = []
    for task_id, cfg in TASK_REGISTRY.items():
        tasks_info.append({
            "task_id": cfg.task_id,
            "difficulty": cfg.difficulty,
            "goal": cfg.goal,
            "max_steps": cfg.max_steps,
            "initial_cash": cfg.initial_cash,
        })
    return tasks_info


@app.post("/grade/{task_id}", response_model=TaskResult)
async def grade_endpoint(task_id: str, episode_trajectory: List[Dict[str, Any]]) -> TaskResult:
    """Accept episode trajectory JSON, return TaskResult."""
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    
    # Delegate to the deterministic pure grader
    return grade_task(task_id, episode_trajectory)


@app.get("/baseline")
async def get_baselines() -> Dict[str, float]:
    """Return pre-computed baseline scores for all 3 tasks.
    
    These are the expected baseline scores for a naive or random agent.
    """
    return {
        "inventory_management": 0.3,
        "supplier_negotiation": 0.2,
        "disruption_response": 0.1,
    }


# Remove existing health route added by create_fastapi_app
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Return basic health and version information."""
    return {
        "status": "ok",
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
