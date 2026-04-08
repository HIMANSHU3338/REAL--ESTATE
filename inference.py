"""
Inference Script — Real Estate RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=portfolio-growth env=real_estate_rl model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=[0,1,0,0,0] reward=0.00 done=false error=null
    [STEP] step=2 action=[0,0,0,3,0] reward=0.01 done=false error=null
    ...
    [END] success=true steps=120 score=0.85 rewards=0.00,0.01,...
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Add project root to path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig, Regime

# ─── Environment Variables (MANDATORY) ───────────────────────────
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: for from_docker_image()

# ─── Task Configuration ─────────────────────────────────────────
TASK_NAME = os.getenv("REAL_ESTATE_TASK", "portfolio-growth")
BENCHMARK = os.getenv("REAL_ESTATE_BENCHMARK", "real_estate_rl")
MAX_STEPS = 120       # 10 years of monthly decisions

# ─── Grading Thresholds ────────────────────────────────────────
# These match the graders in openenv.yaml and server/app.py
TASK_CONFIGS = {
    "portfolio-growth": {
        "success_threshold": 0.3,  # 30%+ return = success
        "description": "Maximize portfolio net worth over 120 months",
    },
    "risk-management": {
        "success_threshold": 0.25,  # Sharpe >= 0.5 = success
        "description": "Maximize risk-adjusted returns (Sharpe ratio)",
    },
    "market-timing": {
        "success_threshold": 0.3,
        "description": "Maximize returns by timing market regime changes",
    },
}


# ─── Stdout Logging (MANDATORY FORMAT) ──────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── Rule-Based Action Generation ─────────────────────────────────

def get_model_action(
    obs: List[float],
    step: int,
) -> List[int]:
    """Generate actions using deterministic rules based on observation."""
    # Get market regime: obs[2:5] is one-hot for BOOM, STABLE, RECESSION
    regime_idx = int(obs[2:5].index(1.0))  # 0=BOOM, 1=STABLE, 2=RECESSION

    actions = []
    for slot in range(5):
        offset = 14 + slot * 11
        occupied = obs[offset] > 0.5

        if not occupied:
            # Empty slot: Buy during RECESSION (regime_idx == 2)
            action = 1 if regime_idx == 2 else 0
        else:
            # Occupied slot
            occupancy = obs[offset + 3]  # Occupancy rate

            if regime_idx == 0:  # BOOM: Sell
                action = 2
            elif occupancy < 0.7:  # Low occupancy: Lower rent
                action = 4
            elif occupancy > 0.9:  # High occupancy: Raise rent
                action = 3
            else:  # Hold
                action = 0

        actions.append(action)

    return actions


# ─── Grading ─────────────────────────────────────────────────────

def compute_score(task_name: str, episode_summary: Dict) -> float:
    """Compute task score in [0, 1] from episode summary."""
    if task_name == "portfolio-growth":
        total_return_pct = episode_summary.get("total_return_pct", 0.0)
        score = max(0.0, min(1.0, total_return_pct / 100.0))

    elif task_name == "risk-management":
        sharpe = episode_summary.get("annualized_sharpe", 0.0)
        score = max(0.0, min(1.0, sharpe / 2.0))

    elif task_name == "market-timing":
        total_return_pct = episode_summary.get("total_return_pct", 0.0)
        max_drawdown_pct = episode_summary.get("max_drawdown_pct", 100.0)
        total_bought = episode_summary.get("total_properties_bought", 0)
        total_sold = episode_summary.get("total_properties_sold", 0)

        return_score = max(0.0, min(0.5, total_return_pct / 200.0))
        dd_score = max(0.0, 0.3 * (1.0 - max_drawdown_pct / 50.0))
        activity_score = min(0.2, (total_bought + total_sold) * 0.02)
        score = max(0.0, min(1.0, return_score + dd_score + activity_score))

    else:
        score = 0.0

    return round(score, 2)


# ─── Main Loop ───────────────────────────────────────────────────

def main() -> None:
    # Initialize environment locally
    config = EnvConfig()
    env = RealEstateEnv(config=config)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_action: Optional[List[int]] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset(seed=42)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            # Get action from rule-based function
            action = get_model_action(obs, step)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(
                __import__("numpy").array(action, dtype=__import__("numpy").int64)
            )

            done = terminated or truncated
            error = None  # Environment doesn't produce action errors in this format
            reward_float = float(reward)

            rewards.append(reward_float)
            steps_taken = step
            last_reward = reward_float
            last_action = action

            # Log step in mandatory format
            action_str = str(action)
            log_step(step=step, action=action_str, reward=reward_float, done=done, error=error)

            # Build history for context window
            regime = Regime.NAMES.get(
                int(obs[2:5].argmax()), "UNKNOWN"
            )
            history.append(
                f"Month {step}: {action_str} -> reward={reward_float:+.4f} | {regime}"
            )

            if done:
                break

        # Compute final score
        episode_summary = info.get("episode_summary", {})
        score = compute_score(TASK_NAME, episode_summary)
        task_config = TASK_CONFIGS.get(TASK_NAME, {"success_threshold": 0.3})
        success = score >= task_config["success_threshold"]

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
