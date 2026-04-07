"""
FastAPI Server — OpenEnv-compatible HTTP endpoints for RealEstateEnv.

Exposes:
    POST /reset   → Reset the environment, return initial observation + info
    POST /step    → Take an action, return obs, reward, done, info
    GET  /state   → Return current environment state snapshot

Required by HF Space deployment and OpenEnv spec compliance.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig


# ─── Pydantic Models (Typed Request/Response) ────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action: List[int] = Field(
        ...,
        description="List of 5 integers (0-4), one action per property slot",
        min_length=5,
        max_length=5,
    )


class ObservationResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    month: int
    cash: float
    net_worth: float
    regime: str
    interest_rate: float
    demand_index: float
    inflation: float
    num_properties: int
    portfolio: Dict[str, Any]
    market: Dict[str, Any]
    episode_progress: float


class TaskInfo(BaseModel):
    name: str
    description: str


class EnvInfoResponse(BaseModel):
    name: str
    version: str
    observation_space_shape: List[int]
    action_space_shape: List[int]
    episode_length: int
    tasks: List[TaskInfo]


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
async def get_info() -> EnvInfoResponse:
    """Return environment metadata and available tasks."""
    return EnvInfoResponse(
        name="real_estate_rl",
        version="1.0.0",
        observation_space_shape=list(env.observation_space.shape),
        action_space_shape=list(env.action_space.nvec),
        episode_length=config.episode_length,
        tasks=[
            TaskInfo(name="portfolio-growth", description="Maximize portfolio net worth over 120 months"),
            TaskInfo(name="risk-management", description="Maximize risk-adjusted returns (Sharpe ratio)"),
            TaskInfo(name="market-timing", description="Maximize returns by timing market regime changes"),
        ],
    )


@app.post("/reset")
async def reset(request: ResetRequest = None) -> ObservationResponse:
    """Reset the environment. Optionally set seed and task."""
    global current_task

    req = request or ResetRequest()

    # Extract task from options if provided
    if req.options and "task" in req.options:
        current_task = req.options["task"]

    obs, info = env.reset(seed=req.seed)

    return ObservationResponse(
        observation=obs.tolist(),
        info=clean_info(info),
    )


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    """Take one step in the environment."""
    action = np.array(request.action, dtype=np.int64)

    # Validate action range
    for i, a in enumerate(request.action):
        if a < 0 or a > 4:
            raise HTTPException(
                status_code=400,
                detail=f"Action for slot {i} must be in [0, 4], got {a}",
            )

    obs, reward, terminated, truncated, info = env.step(action)

    # If episode is done, include grading score
    if (terminated or truncated) and current_task:
        info["task_score"] = _grade_task(current_task, info)

    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        done=bool(terminated or truncated),
        info=clean_info(info),
    )


@app.get("/state")
async def get_state() -> StateResponse:
    """Return the current environment state."""
    state = env.state()

    # Convert numpy types
    clean_state = clean_info(state)

    return StateResponse(**clean_state)


# ─── Grading Logic ───────────────────────────────────────────────

def _grade_task(task_name: str, final_info: Dict) -> float:
    """
    Grade a completed episode for a specific task.
    Returns score in [0.0, 1.0].
    """
    summary = final_info.get("episode_summary", {})

    if task_name == "portfolio-growth":
        return _grade_portfolio_growth(summary)
    elif task_name == "risk-management":
        return _grade_risk_management(summary)
    elif task_name == "market-timing":
        return _grade_market_timing(summary)
    else:
        return 0.0


def _grade_portfolio_growth(summary: Dict) -> float:
    """
    Portfolio Growth Task:
    Score based on total return percentage.
    Target: 100%+ return over 10 years = 1.0
    Baseline: 0% return = 0.0, negative = 0.0
    """
    total_return_pct = summary.get("total_return_pct", 0.0)
    # Map: -inf..0 -> 0.0, 0..100 -> 0.0..1.0, 100+ -> 1.0
    score = max(0.0, min(1.0, total_return_pct / 100.0))
    return round(score, 4)


def _grade_risk_management(summary: Dict) -> float:
    """
    Risk Management Task:
    Score based on Sharpe ratio.
    Target: Sharpe >= 2.0 = 1.0
    Baseline: Sharpe <= 0 = 0.0
    """
    sharpe = summary.get("annualized_sharpe", 0.0)
    # Map: <=0 -> 0.0, 0..2 -> 0.0..1.0, 2+ -> 1.0
    score = max(0.0, min(1.0, sharpe / 2.0))
    return round(score, 4)


def _grade_market_timing(summary: Dict) -> float:
    """
    Market Timing Task:
    Score based on combination of:
    - Total return (did you profit overall?)
    - Max drawdown (did you avoid big losses?)
    - Number of transactions (active management)
    """
    total_return_pct = summary.get("total_return_pct", 0.0)
    max_drawdown_pct = summary.get("max_drawdown_pct", 100.0)
    total_bought = summary.get("total_properties_bought", 0)
    total_sold = summary.get("total_properties_sold", 0)

    # Return component (0-0.5)
    return_score = max(0.0, min(0.5, total_return_pct / 200.0))

    # Drawdown component: lower is better (0-0.3)
    dd_score = max(0.0, 0.3 * (1.0 - max_drawdown_pct / 50.0))

    # Activity component: some trading is needed (0-0.2)
    transactions = total_bought + total_sold
    activity_score = min(0.2, transactions * 0.02)

    score = max(0.0, min(1.0, return_score + dd_score + activity_score))
    return round(score, 4)


# ─── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
