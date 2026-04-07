"""
Training Script — Train an RL agent on the Real Estate Environment.

Uses Stable-Baselines3 PPO (Proximal Policy Optimization).
Includes:
- Hyperparameter configuration optimized for financial environments
- Learning rate scheduling
- TensorBoard logging
- Model checkpointing
- Training progress display
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with:")
    print("    pip install stable-baselines3[extra]")


def linear_schedule(initial_lr: float):
    """Linear learning rate decay."""
    def schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return schedule


def make_env(config: EnvConfig, seed: int = 0):
    """Create an environment factory."""
    def _init():
        env = RealEstateEnv(config=config)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_dir: str = "trained_models",
    log_dir: str = "logs",
    seed: int = 42,
    resume_from: str = None,
):
    """
    Train a PPO agent on the Real Estate Environment.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save checkpoints
        log_dir: TensorBoard log directory
        seed: Random seed
        resume_from: Path to a saved model to resume from
    """
    if not HAS_SB3:
        print("❌ Cannot train without stable-baselines3. Exiting.")
        return None
    
    print("🏠 Real Estate RL — Training Pipeline")
    print("=" * 50)
    
    config = EnvConfig()
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environments
    print(f"🔧 Creating {n_envs} parallel environments...")
    env = DummyVecEnv([make_env(config, seed=seed + i) for i in range(n_envs)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(config, seed=seed + 100)])
    
    # Model setup
    if resume_from and os.path.exists(resume_from):
        print(f"📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("🧠 Initializing PPO agent...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_schedule(3e-4),
            n_steps=2048,           # Steps per rollout
            batch_size=256,         # Minibatch size
            n_epochs=10,            # Epochs per update
            gamma=0.99,             # Discount factor
            gae_lambda=0.95,        # GAE lambda
            clip_range=0.2,         # PPO clip range
            clip_range_vf=None,     # No value function clipping
            ent_coef=0.01,          # Entropy bonus (encourage exploration)
            vf_coef=0.5,            # Value function coefficient
            max_grad_norm=0.5,      # Gradient clipping
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256, 128],   # Policy network
                    vf=[256, 256, 128],   # Value network
                ),
            ),
            tensorboard_log=log_dir,
            seed=seed,
            verbose=1,
        )
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="real_estate_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=log_dir,
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Training
    print(f"\n🚀 Training for {total_timesteps:,} timesteps...")
    print(f"   📊 TensorBoard: tensorboard --logdir {log_dir}")
    print(f"   💾 Checkpoints: {save_dir}/")
    print()
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Save final model
    final_path = os.path.join(save_dir, "real_estate_ppo_final")
    model.save(final_path)
    
    print(f"\n✅ Training complete!")
    print(f"   ⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   💾 Final model: {final_path}")
    print(f"   🏆 Best model: {save_dir}/best/best_model")
    
    # Save training config
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "algorithm": "PPO",
            "policy": "MlpPolicy",
        }, f, indent=2)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Real Estate RL Agent")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="trained_models", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        resume_from=args.resume,
    )
