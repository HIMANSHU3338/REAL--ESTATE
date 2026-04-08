---
title: Real Estate Investment RL
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🏠 Real Estate Investment RL Environment

A production-quality **Reinforcement Learning environment** for real estate portfolio management, built on **Gymnasium** and fully compliant with the **OpenEnv** submission spec.

## 🌟 What Makes This Different

| Feature | Typical Projects | This Project |
|---------|-----------------|-------------|
| Properties | Single property | **Portfolio of up to 5** |
| Market | Static/random | **Regime switching (boom/bust/stable)** |
| Financing | Cash only | **Leverage + mortgage modeling** |
| Neighborhoods | None | **3 distinct areas with dynamics** |
| Reward | Simple profit | **Sharpe Ratio (risk-adjusted)** |
| Baselines | None | **3 baseline agents for benchmarking** |
| Visualization | Terminal output | **Web dashboard** |
| Deployment | Local only | **HF Space + Docker + OpenEnv** |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test
python main.py --mode test

# Run demo episode
python main.py --mode demo

# Evaluate all agents
python main.py --mode evaluate

# Train PPO agent
python main.py --mode train

# Full pipeline
python main.py --mode all
```

## 📦 OpenEnv Submission

### Required Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"  # or your endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"           # or your model
export HF_TOKEN="your-hugging-face-token"                 # required, no default
# Optional:
export LOCAL_IMAGE_NAME="your-docker-image"               # for from_docker_image()
```

### Run Inference Script

```bash
python inference.py
```

This will output the required `[START]`, `[STEP]`, `[END]` format to stdout.

### Run Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The server exposes:
- `POST /reset` — Reset environment
- `POST /step` — Take an action
- `GET /state` — Get current state
- `GET /info` — Environment metadata & tasks

### Docker Build

```bash
docker build -t real-estate-rl .
docker run -p 7860:7860 real-estate-rl
```

### Pre-Submission Validation

```bash
pip install openenv-core
./validate-submission.sh https://your-space.hf.space .
```

### Available Tasks

| Task | Description | Scoring |
|------|-------------|---------|
| `portfolio-growth` | Maximize net worth over 120 months | `score = return% / 100` |
| `risk-management` | Maximize Sharpe ratio | `score = sharpe / 2.0` |
| `market-timing` | Time buy/sell around regime changes | Composite (return + drawdown + activity) |

## 🏗️ Architecture

```
real_estate_rl/
├── env/                    # Gymnasium Environment
│   ├── config.py           # All hyperparameters
│   ├── market_engine.py    # Market simulation (regimes, rates, demand)
│   ├── property_manager.py # Portfolio management
│   └── real_estate_env.py  # Core env (step/reset/render/state)
├── agents/                 # RL Agents
│   ├── baselines.py        # Random, BuyAndHold, RuleBased
│   ├── train.py            # PPO training pipeline
│   └── evaluate.py         # Benchmarking & comparison
├── server/                 # OpenEnv HTTP Server
│   ├── app.py              # FastAPI endpoints (/reset, /step, /state)
│   └── Dockerfile          # Server Docker config
├── utils/                  # Utilities
│   ├── logger.py           # Episode logging
│   └── plotting.py         # Matplotlib charts
├── dashboard/              # Web visualization
│   ├── index.html
│   ├── style.css
│   └── app.js
├── inference.py            # Root inference script (MANDATORY)
├── openenv.yaml            # OpenEnv spec (tasks, models, graders)
├── Dockerfile              # Root Docker build
├── main.py                 # Entry point
└── requirements.txt
```

## 🧠 Environment Details

### State Space (69 dimensions)
- **Global**: Cash, net worth, market regime, interest rate, demand, inflation, seasonality
- **Per Property (×5)**: Value, rent, occupancy, mortgage, neighborhood, type

### Action Space (MultiDiscrete [5,5,5,5,5])
Per property slot: Hold | Buy | Sell | Raise Rent | Lower Rent

### Reward
```
reward = 0.6 × step_return + 0.4 × rolling_sharpe - penalties
```

### Market Regimes
- **BOOM**: Prices ↑, demand high, rates rising
- **STABLE**: Moderate growth, balanced demand
- **RECESSION**: Prices ↓, demand low, rates falling

## 📊 Baselines

1. **RandomAgent**: Random actions (lower bound)
2. **BuyAndHoldAgent**: Passive investor, never sells
3. **RuleBasedAgent**: Smart heuristics (buy low, sell high)
4. **PPO Trained**: RL agent trained with Stable-Baselines3

## 📈 Training

Uses **PPO** (Proximal Policy Optimization) with:
- Linear learning rate decay
- 256-256-128 network architecture
- Entropy bonus for exploration
- 4 parallel environments
- Checkpointing + TensorBoard logging

## 📜 License

MIT
