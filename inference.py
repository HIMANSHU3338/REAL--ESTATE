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

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─── Add project root to path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig, Regime

# ─── Environment Variables (MANDATORY) ───────────────────────────
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: for from_docker_image()

# Use injected environment variables for API access
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ─── Task Configuration ─────────────────────────────────────────
TASK_NAME = os.getenv("REAL_ESTATE_TASK", "portfolio-growth")
BENCHMARK = os.getenv("REAL_ESTATE_BENCHMARK", "real_estate_rl")
MAX_STEPS = 120       # 10 years of monthly decisions
TEMPERATURE = 0.7
MAX_TOKENS = 300

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

# ─── System Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert real estate investment AI managing a portfolio in the Indian real estate market.

    ENVIRONMENT:
    - You manage up to 5 property slots across 3 neighborhoods (South Mumbai, Bandra, Navi Mumbai).
    - Each month you choose one action per slot: 0=Hold, 1=Buy, 2=Sell, 3=Raise Rent, 4=Lower Rent.
    - The market cycles through BOOM, STABLE, and RECESSION regimes.
    - Starting capital: ₹2 Crore (₹20,000,000).

    STRATEGY GUIDELINES:
    - Buy during RECESSION (low prices), sell during BOOM (high prices).
    - Adjust rent based on occupancy: lower rent if occupancy < 70%, raise if > 90%.
    - Keep cash reserves for opportunities and to avoid foreclosure.
    - Diversify across neighborhoods.
    - Monitor interest rates — high rates mean expensive mortgages.

    RESPONSE FORMAT:
    You must respond with ONLY a JSON array of exactly 5 integers, each in [0,4].
    Example: [0, 1, 0, 3, 0]
    No explanation, no extra text — just the JSON array.
""").strip()


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


# ─── Observation Parsing ─────────────────────────────────────────

def parse_observation(obs, step_count: int, config: EnvConfig) -> Dict[str, Any]:
    """Parse the 69-dim observation vector into a human-readable dict for the LLM."""
    result = {}

    # Global features (indices 0-13)
    result["cash_level"] = f"{obs[0]:.2%}"  # normalized
    result["net_worth_level"] = f"{obs[1]:.2%}"  # normalized

    # Regime one-hot (indices 2-4)
    regime_idx = int(obs[2:5].argmax())
    result["market_regime"] = Regime.NAMES[regime_idx]

    # Market features
    result["interest_rate_norm"] = f"{obs[5]:.2f}"
    result["demand"] = f"{obs[6]:.1%}"
    result["inflation_norm"] = f"{obs[7]:.2f}"

    # Time
    result["month"] = step_count
    result["time_remaining"] = f"{obs[10]:.1%}"

    # Property slots (5 slots, 11 features each, starting at index 14)
    properties = []
    for slot in range(5):
        offset = 14 + slot * 11
        occupied = obs[offset] > 0.5

        if occupied:
            properties.append({
                "slot": slot,
                "occupied": True,
                "value_norm": f"{obs[offset + 1]:.3f}",
                "rent_norm": f"{obs[offset + 2]:.3f}",
                "occupancy": f"{obs[offset + 3]:.0%}",
                "mortgage_norm": f"{obs[offset + 4]:.3f}",
            })
        else:
            properties.append({"slot": slot, "occupied": False})

    result["properties"] = properties
    return result


def build_user_prompt(
    step: int,
    obs_info: Dict,
    last_reward: float,
    last_action: Optional[List[int]],
    history: List[str],
) -> str:
    """Build a concise prompt for the LLM with current state."""
    props_str = ""
    for p in obs_info["properties"]:
        if p["occupied"]:
            props_str += (
                f"  Slot {p['slot']}: OCCUPIED | Value={p['value_norm']} | "
                f"Rent={p['rent_norm']} | Occupancy={p['occupancy']} | "
                f"Mortgage={p['mortgage_norm']}\n"
            )
        else:
            props_str += f"  Slot {p['slot']}: EMPTY\n"

    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""
        Month: {step}/{MAX_STEPS}
        Market Regime: {obs_info['market_regime']}
        Cash Level: {obs_info['cash_level']}
        Net Worth Level: {obs_info['net_worth_level']}
        Demand: {obs_info['demand']}
        Interest Rate: {obs_info['interest_rate_norm']}
        Time Remaining: {obs_info['time_remaining']}

        Properties:
        {props_str}
        Last Action: {last_action}
        Last Reward: {last_reward:.4f}

        Recent History:
        {history_block}

        Choose your next action — respond with ONLY a JSON array of 5 integers [0-4].
    """).strip()


# ─── LLM Interaction ─────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    step: int,
    obs_info: Dict,
    last_reward: float,
    last_action: Optional[List[int]],
    history: List[str],
) -> List[int]:
    """Query the LLM for the next action. Returns list of 5 ints."""
    user_prompt = build_user_prompt(step, obs_info, last_reward, last_action, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse the JSON array from the response
        action = _parse_action(text)
        return action

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return [0, 0, 0, 0, 0]  # Default: Hold everything


def _parse_action(text: str) -> List[int]:
    """Parse LLM response into a valid action array."""
    # Try direct JSON parse
    try:
        # Extract JSON array if embedded in text
        import re
        match = re.search(r'\[[\s\d,]+\]', text)
        if match:
            action = json.loads(match.group())
            if isinstance(action, list) and len(action) == 5:
                return [max(0, min(4, int(a))) for a in action]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to find 5 numbers
    try:
        import re
        numbers = re.findall(r'\d', text)
        if len(numbers) >= 5:
            return [max(0, min(4, int(n))) for n in numbers[:5]]
    except Exception:
        pass

    # Ultimate fallback: Hold everything
    return [0, 0, 0, 0, 0]


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

async def main() -> None:
    # Initialize OpenAI client with injected credentials
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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
            # Parse observation for the LLM
            obs_info = parse_observation(obs, step, config)

            # Get action from LLM
            action = get_model_action(
                client, step, obs_info, last_reward, last_action, history
            )

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
    asyncio.run(main())
