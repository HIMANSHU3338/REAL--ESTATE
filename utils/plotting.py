"""
Plotting Utilities — Training curves and evaluation charts.

Uses matplotlib to generate publication-quality plots.
"""

import os
import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def set_style():
    """Set a clean, modern plot style."""
    if not HAS_MPL:
        return
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 120,
    })


def plot_training_rewards(rewards: List[float], window: int = 50, save_path: str = None):
    """Plot training reward curve with moving average."""
    if not HAS_MPL:
        print("⚠️  matplotlib not installed, skipping plot")
        return
    
    set_style()
    fig, ax = plt.subplots()
    
    episodes = range(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color='#4ECDC4', linewidth=0.5, label='Episode Reward')
    
    # Moving average
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), ma, color='#FF6B6B', linewidth=2,
                label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('🏠 Real Estate RL — Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Training plot saved to {save_path}")
    plt.close()


def plot_net_worth_comparison(results: Dict, save_path: str = None):
    """Plot net worth histories for different agents side by side."""
    if not HAS_MPL:
        return
    
    set_style()
    colors = {'Random': '#95a5a6', 'BuyAndHold': '#3498db', 'RuleBased': '#f39c12', 'PPO_Trained': '#e74c3c'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        
        episodes = result.get("episode_details", [])
        for ep in episodes[:20]:  # Plot first 20 episodes
            nw = ep.get("net_worth_history", [])
            if nw:
                ax.plot(range(len(nw)), nw, alpha=0.2, color=colors.get(name, '#333'))
        
        # Plot average
        all_nw = [ep.get("net_worth_history", []) for ep in episodes]
        if all_nw:
            max_len = max(len(nw) for nw in all_nw)
            padded = [nw + [nw[-1]] * (max_len - len(nw)) if nw else [20_000_000]*max_len for nw in all_nw]
            avg_nw = np.mean(padded, axis=0)
            ax.plot(range(len(avg_nw)), avg_nw, color=colors.get(name, '#333'),
                    linewidth=2.5, label=f'Average')
        
        ax.axhline(y=20_000_000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title(f'{name}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Net Worth (\u20b9)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(results), 4):
        axes[idx].set_visible(False)
    
    fig.suptitle('🏠 Net Worth Trajectories by Agent', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Comparison plot saved to {save_path}")
    plt.close()


def plot_action_distribution(results: Dict, save_path: str = None):
    """Plot action distribution for each agent."""
    if not HAS_MPL:
        return
    
    set_style()
    action_names = ['Hold', 'Buy', 'Sell', 'Raise Rent', 'Lower Rent']
    colors_actions = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    agent_names = list(results.keys())
    n_agents = len(agent_names)
    
    fig, axes = plt.subplots(1, n_agents, figsize=(4 * n_agents, 5))
    if n_agents == 1:
        axes = [axes]
    
    for idx, name in enumerate(agent_names):
        ax = axes[idx]
        dist = results[name].get("action_distribution", {})
        
        values = [dist.get(a, 0) for a in action_names]
        bars = ax.bar(range(len(action_names)), values, color=colors_actions, edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('% of Actions')
        ax.set_title(name, fontweight='bold')
        ax.set_ylim(0, max(values + [50]) * 1.2)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('🎮 Action Distribution by Agent', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"🎮 Action distribution plot saved to {save_path}")
    plt.close()


def plot_metrics_comparison(results: Dict, save_path: str = None):
    """Plot bar chart comparing key metrics across agents."""
    if not HAS_MPL:
        return
    
    set_style()
    agents = list(results.keys())
    metrics = {
        'Avg Return (%)': [results[a]['avg_return_pct'] for a in agents],
        'Sharpe Ratio': [results[a]['avg_sharpe'] for a in agents],
        'Win Rate (%)': [results[a]['win_rate_pct'] for a in agents],
    }
    
    x = np.arange(len(agents))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=colors[i],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Value')
    ax.set_title('🏆 Agent Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(agents)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"🏆 Metrics comparison saved to {save_path}")
    plt.close()
