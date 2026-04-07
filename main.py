"""
🏠 Real Estate Investment RL — Main Entry Point

Usage:
    python main.py --mode train       # Train the PPO agent
    python main.py --mode evaluate    # Evaluate all agents
    python main.py --mode demo        # Run a demo episode
    python main.py --mode test        # Quick environment test
    python main.py --mode all         # Full pipeline
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig, Regime
from agents.baselines import RandomAgent, BuyAndHoldAgent, RuleBasedAgent
from utils.logger import EpisodeLogger


def test_environment():
    """Quick test: validate environment works correctly."""
    print("\n🧪 ENVIRONMENT TEST")
    print("=" * 60)
    
    config = EnvConfig()
    env = RealEstateEnv(config=config)
    
    # 1. Check env with gymnasium checker
    print("1️⃣  Running gymnasium env checker...")
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, warn=True)
        print("   ✅ Environment passed all checks!")
    except Exception as e:
        print(f"   ⚠️  Checker warning: {e}")
    
    # 2. Run random episodes
    print("\n2️⃣  Running 5 random episodes...")
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Check for NaN
            if np.any(np.isnan(obs)):
                print(f"   ❌ NaN in observation at step {steps}!")
                break
            if np.isnan(reward):
                print(f"   ❌ NaN reward at step {steps}!")
                break
            
            if terminated or truncated:
                break
        
        summary = info.get("episode_summary", {})
        final_nw = summary.get("final_net_worth", 0)
        ret = summary.get("total_return_pct", 0)
        print(f"   Episode {ep+1}: {steps} steps, "
              f"Reward={total_reward:.3f}, "
              f"NW=\u20b9{final_nw:,}, "
              f"Return={ret:.1f}%")
    
    # 3. Test observation/action spaces
    print(f"\n3️⃣  Space dimensions:")
    print(f"   Observation: {env.observation_space.shape}")
    print(f"   Action: {env.action_space.nvec}")
    
    # 4. Test baselines
    print(f"\n4️⃣  Quick baseline test (1 episode each)...")
    baselines = {
        "Random": RandomAgent(num_slots=config.max_properties, seed=42),
        "BuyAndHold": BuyAndHoldAgent(num_slots=config.max_properties),
        "RuleBased": RuleBasedAgent(num_slots=config.max_properties),
    }
    
    for name, agent in baselines.items():
        obs, info = env.reset(seed=42)
        total_reward = 0
        done = False
        while not done:
            action = agent.predict(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        summary = info.get("episode_summary", {})
        print(f"   {name:<12}: NW=\u20b9{summary.get('final_net_worth', 0):>10,} | "
              f"Return={summary.get('total_return_pct', 0):>6.1f}% | "
              f"Sharpe={summary.get('annualized_sharpe', 0):>6.3f}")
    
    print(f"\n✅ All tests passed!")
    print("=" * 60)


def run_demo():
    """Run a visual demo episode."""
    from agents.evaluate import run_demo_episode
    
    config = EnvConfig()
    agent = RuleBasedAgent(num_slots=config.max_properties)
    
    demo_data = run_demo_episode(agent, "RuleBased", config, seed=42)
    
    # Save for dashboard
    os.makedirs("results", exist_ok=True)
    with open("results/demo_episode.json", "w") as f:
        json.dump(demo_data, f, indent=2, default=str)
    
    print(f"\n💾 Demo data saved to results/demo_episode.json")
    return demo_data


def run_evaluation():
    """Run full evaluation with all agents."""
    from agents.evaluate import evaluate_all
    from utils.plotting import (
        plot_net_worth_comparison,
        plot_action_distribution,
        plot_metrics_comparison,
    )
    
    results = evaluate_all(
        model_path="trained_models/real_estate_ppo_final",
        n_episodes=50,
        output_dir="results",
    )
    
    # Generate plots
    print("\n📊 Generating plots...")
    os.makedirs("results/plots", exist_ok=True)
    
    plot_net_worth_comparison(results, save_path="results/plots/net_worth_comparison.png")
    plot_action_distribution(results, save_path="results/plots/action_distribution.png")
    plot_metrics_comparison(results, save_path="results/plots/metrics_comparison.png")
    
    return results


def run_training():
    """Run the training pipeline."""
    from agents.train import train
    
    model = train(
        total_timesteps=500_000,
        n_envs=4,
        save_dir="trained_models",
        log_dir="logs",
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="🏠 Real Estate Investment RL Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode test       Quick environment validation
  python main.py --mode demo       Run a visual demo episode
  python main.py --mode evaluate   Evaluate all agents
  python main.py --mode train      Train PPO agent (requires stable-baselines3)
  python main.py --mode all        Full pipeline
        """
    )
    parser.add_argument(
        "--mode", type=str, default="test",
        choices=["test", "demo", "evaluate", "train", "all"],
        help="Run mode (default: test)"
    )
    
    args = parser.parse_args()
    
    print("🏠" + "=" * 58)
    print("   REAL ESTATE INVESTMENT RL ENVIRONMENT")
    print("   Multi-Property Portfolio • Market Cycles • Sharpe Reward")
    print("=" * 60)
    
    if args.mode == "test":
        test_environment()
    
    elif args.mode == "demo":
        test_environment()
        run_demo()
    
    elif args.mode == "evaluate":
        run_evaluation()
    
    elif args.mode == "train":
        run_training()
    
    elif args.mode == "all":
        test_environment()
        print("\n" + "🚀" * 20)
        run_training()
        print("\n" + "📊" * 20)
        run_evaluation()
        print("\n" + "🎬" * 20)
        run_demo()


if __name__ == "__main__":
    main()
