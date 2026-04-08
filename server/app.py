"""
FastAPI Server — OpenEnv-compatible HTTP endpoints for RealEstateEnv.

Uses minimal request/response handling for maximum compatibility with OpenEnv validation.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig


# ─── App Setup ────────────────────────────────────────────────────

app = FastAPI(
    title="Real Estate RL — OpenEnv Server",
    description="OpenEnv-compatible environment for real estate portfolio management.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
config = EnvConfig()
env = RealEstateEnv(config=config)

# Track current task for grading
current_task: Optional[str] = None


def numpy_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_info(info: Dict) -> Dict:
    """Recursively convert numpy types in info dict."""
    cleaned = {}
    for k, v in info.items():
        if isinstance(v, dict):
            cleaned[k] = clean_info(v)
        elif isinstance(v, list):
            cleaned[k] = [
                clean_info(item) if isinstance(item, dict) else numpy_serializable(item)
                for item in v
            ]
        elif isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        else:
            cleaned[k] = numpy_serializable(v)
    return cleaned


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "env": "real_estate_rl", "version": "1.0.0"}


@app.get("/info")
async def get_info() -> Dict:
    """Return environment metadata and available tasks."""
    return {
        "name": "real_estate_rl",
        "version": "1.0.0",
        "observation_space_shape": [int(x) for x in env.observation_space.shape],
        "action_space_shape": [int(x) for x in env.action_space.nvec],
        "episode_length": int(config.episode_length),
        "tasks": [
            {"name": "portfolio-growth", "description": "Maximize portfolio net worth over 120 months"},
            {"name": "risk-management", "description": "Maximize risk-adjusted returns (Sharpe ratio)"},
            {"name": "market-timing", "description": "Maximize returns by timing market regime changes"},
        ],
    }


@app.post("/reset")
async def reset(request: Request):
    """Reset the environment. Optionally set seed and task."""
    global current_task
    
    try:
        data = await request.json()
    except:
        data = {}
    
    seed = data.get("seed") if isinstance(data, dict) else None
    options = data.get("options") if isinstance(data, dict) else None
    
    # Extract task from options if provided
    if options and isinstance(options, dict) and "task" in options:
        current_task = options["task"]
    
    obs, info = env.reset(seed=seed)
    
    return {
        "observation": obs.tolist(),
        "info": clean_info(info),
    }


@app.post("/step")
async def step(request: Request):
    """Take one step in the environment."""
    try:
        data = await request.json()
    except:
        data = {}
    
    # Extract action from request
    action = data.get("action") if isinstance(data, dict) else []
    
    if not action or len(action) != 5:
        raise HTTPException(
            status_code=400,
            detail="action must be a list of exactly 5 integers in [0, 4]",
        )
    
    action_array = np.array(action, dtype=np.int64)
    
    # Validate action range
    for i, a in enumerate(action):
        if not isinstance(a, int) or a < 0 or a > 4:
            raise HTTPException(
                status_code=400,
                detail=f"Action for slot {i} must be in [0, 4], got {a}",
            )
    
    obs, reward, terminated, truncated, info = env.step(action_array)
    
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": clean_info(info),
    }


@app.get("/state")
async def get_state() -> Dict:
    """Return the current environment state."""
    state = env.state()
    return clean_info(state)


# ─── Server Entry Point ───────────────────────────────────────────

def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
