"""
Evaluation Script — Benchmark the trained RL agent against baselines.

Runs multiple episodes and computes:
- Total Return
- Annualized Return
- Sharpe Ratio
- Max Drawdown
- Win Rate (vs initial capital)
- Action distribution

Outputs comparison table and saves results for the dashboard.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig
from agents.baselines import RandomAgent, BuyAndHoldAgent, RuleBasedAgent

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def run_episode(env: RealEstateEnv, agent, seed: int = None) -> Dict:
    """Run a single episode and return summary."""
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    done = False
    actions_taken = []
    
    while not done:
        if hasattr(agent, 'predict') and hasattr(agent, 'policy'):
            # SB3 model
            action, _ = agent.predict(obs, deterministic=True)
        else:
            # Baseline agent
            action = agent.predict(obs, info)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if "step_detail" in info:
            for a in info["step_detail"]["actions"]:
                actions_taken.append(a["action"])
    
    summary = info.get("episode_summary", {})
    summary["total_reward"] = round(total_reward, 4)
    summary["actions_taken"] = actions_taken
    
    return summary


def evaluate_agent(
    agent,
    agent_name: str,
    config: EnvConfig,
    n_episodes: int = 50,
    base_seed: int = 1000,
) -> Dict:
    """Evaluate an agent over multiple episodes."""
    env = RealEstateEnv(config=config)
    
    results = []
    for ep in range(n_episodes):
        summary = run_episode(env, agent, seed=base_seed + ep)
        results.append(summary)
    
    # Aggregate metrics
    returns = [r.get("total_return_pct", 0) for r in results]
    sharpes = [r.get("annualized_sharpe", 0) for r in results]
    drawdowns = [r.get("max_drawdown_pct", 0) for r in results]
    rewards = [r.get("total_reward", 0) for r in results]
    final_nw = [r.get("final_net_worth", 0) for r in results]
    
    # Action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r.get("actions_taken", []))
    
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    total_actions = max(len(all_actions), 1)
    action_dist = {k: round(v / total_actions * 100, 1) for k, v in action_counts.items()}
    
    win_rate = sum(1 for r in returns if r > 0) / max(len(returns), 1) * 100
    
    return {
        "agent": agent_name,
        "n_episodes": n_episodes,
        "avg_return_pct": round(np.mean(returns), 2),
        "std_return_pct": round(np.std(returns), 2),
        "avg_sharpe": round(np.mean(sharpes), 3),
        "avg_max_drawdown_pct": round(np.mean(drawdowns), 2),
        "avg_reward": round(np.mean(rewards), 4),
        "avg_final_net_worth": round(np.mean(final_nw)),
        "win_rate_pct": round(win_rate, 1),
        "action_distribution": action_dist,
        "best_return_pct": round(max(returns), 2),
        "worst_return_pct": round(min(returns), 2),
        "episode_details": results,
    }


def run_demo_episode(agent, agent_name: str, config: EnvConfig, seed: int = 42) -> Dict:
    """Run one episode with full logging for dashboard visualization."""
    env = RealEstateEnv(config=config, render_mode="human")
    obs, info = env.reset(seed=seed)
    
    print(f"\n{'🎬 DEMO EPISODE':=^60}")
    print(f"  Agent: {agent_name}")
    print(f"  Seed: {seed}")
    print(f"  Episode Length: {config.episode_length} months")
    print(f"{'='*60}\n")
    
    total_reward = 0.0
    done = False
    
    while not done:
        if hasattr(agent, 'predict') and hasattr(agent, 'policy'):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs, info)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        env.render()
    
    summary = info.get("episode_summary", {})
    summary["total_reward"] = total_reward
    
    print(f"\n{'📊 EPISODE SUMMARY':=^60}")
    print(f"  Final Net Worth: \u20b9{summary.get('final_net_worth', 0):,}")
    print(f"  Total Return: {summary.get('total_return_pct', 0):.1f}%")
    print(f"  Annualized Return: {summary.get('annualized_return_pct', 0):.1f}%")
    print(f"  Sharpe Ratio: {summary.get('annualized_sharpe', 0):.3f}")
    print(f"  Max Drawdown: {summary.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Properties Bought: {summary.get('total_properties_bought', 0)}")
    print(f"  Properties Sold: {summary.get('total_properties_sold', 0)}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"{'='*60}")
    
    # Save episode log for dashboard
    episode_log = env.get_episode_log()
    
    return {
        "summary": summary,
        "log": episode_log,
        "net_worth_history": summary.get("net_worth_history", []),
        "regime_history": summary.get("regime_history", []),
    }


def evaluate_all(
    model_path: str = None,
    n_episodes: int = 50,
    output_dir: str = "results",
):
    """Run full evaluation: trained agent + all baselines."""
    print("🏠 Real Estate RL — Evaluation Pipeline")
    print("=" * 50)
    
    config = EnvConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Setup agents ──
    agents = {
        "Random": RandomAgent(num_slots=config.max_properties),
        "BuyAndHold": BuyAndHoldAgent(num_slots=config.max_properties),
        "RuleBased": RuleBasedAgent(num_slots=config.max_properties),
    }
    
    # Load trained model if available
    if model_path and HAS_SB3:
        if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
            print(f"📂 Loading trained model: {model_path}")
            agents["PPO_Trained"] = PPO.load(model_path)
        else:
            print(f"⚠️  Model not found: {model_path}")
    elif not HAS_SB3:
        print("⚠️  stable-baselines3 not installed, skipping trained model")
    
    # ── Evaluate each agent ──
    all_results = {}
    for name, agent in agents.items():
        print(f"\n🔄 Evaluating {name}... ({n_episodes} episodes)")
        result = evaluate_agent(agent, name, config, n_episodes=n_episodes)
        all_results[name] = result
        
        print(f"   📈 Avg Return: {result['avg_return_pct']:.1f}% ± {result['std_return_pct']:.1f}%")
        print(f"   📊 Avg Sharpe: {result['avg_sharpe']:.3f}")
        print(f"   📉 Avg Max DD: {result['avg_max_drawdown_pct']:.1f}%")
        print(f"   🏆 Win Rate: {result['win_rate_pct']:.0f}%")
    
    # ── Print comparison table ──
    print(f"\n{'='*80}")
    print(f"{'COMPARISON TABLE':^80}")
    print(f"{'='*80}")
    print(f"{'Agent':<15} {'Avg Return%':>12} {'Sharpe':>10} {'Max DD%':>10} {'Win Rate%':>10} {'Avg NW':>12}")
    print(f"{'-'*80}")
    
    for name, r in all_results.items():
        print(f"{name:<15} {r['avg_return_pct']:>11.1f}% {r['avg_sharpe']:>10.3f} "
              f"{r['avg_max_drawdown_pct']:>9.1f}% {r['win_rate_pct']:>9.0f}% "
              f"\u20b9{r['avg_final_net_worth']:>10,}")
    print(f"{'='*80}")
    
    # ── Save results ──
    # Remove episode_details for the summary file (too large)
    summary_results = {}
    for name, r in all_results.items():
        summary_results[name] = {k: v for k, v in r.items() if k != "episode_details"}
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_path}")
    
    # ── Run demo episode with best agent ──
    best_agent_name = max(all_results, key=lambda k: all_results[k]["avg_return_pct"])
    best_agent = agents[best_agent_name]
    
    print(f"\n🎬 Running demo episode with best agent: {best_agent_name}")
    demo = run_demo_episode(best_agent, best_agent_name, config)
    
    demo_path = os.path.join(output_dir, "demo_episode.json")
    with open(demo_path, "w") as f:
        json.dump(demo, f, indent=2, default=str)
    print(f"💾 Demo saved to: {demo_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Real Estate RL Agents")
    parser.add_argument("--model", type=str, default="trained_models/real_estate_ppo_final",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per agent")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    evaluate_all(model_path=args.model, n_episodes=args.episodes, output_dir=args.output)
