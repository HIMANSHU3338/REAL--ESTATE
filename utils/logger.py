"""
Logger — Episode logging and metrics tracking.

Saves episode data as JSON for dashboard consumption.
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime


class EpisodeLogger:
    """Logs episode data to JSON files."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.episodes = []
    
    def log_episode(self, summary: Dict, episode_log: List[Dict] = None):
        """Log a completed episode."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "episode_num": len(self.episodes) + 1,
            "summary": summary,
        }
        if episode_log:
            entry["steps"] = episode_log
        
        self.episodes.append(entry)
    
    def save(self, filename: str = "episode_logs.json"):
        """Save all logged episodes to JSON."""
        path = os.path.join(self.output_dir, filename)
        
        # Make data JSON-serializable
        def serialize(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)
        
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2, default=serialize)
        
        print(f"💾 Saved {len(self.episodes)} episodes to {path}")
        return path
    
    def save_dashboard_data(self, evaluation_results: Dict, demo_data: Dict = None):
        """
        Save data formatted specifically for the dashboard.
        Combines evaluation results and demo episode data.
        """
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "evaluation": {},
            "demo_episode": None,
        }
        
        # Process evaluation results
        for agent_name, result in evaluation_results.items():
            dashboard_data["evaluation"][agent_name] = {
                "avg_return_pct": result.get("avg_return_pct", 0),
                "std_return_pct": result.get("std_return_pct", 0),
                "avg_sharpe": result.get("avg_sharpe", 0),
                "avg_max_drawdown_pct": result.get("avg_max_drawdown_pct", 0),
                "win_rate_pct": result.get("win_rate_pct", 0),
                "avg_final_net_worth": result.get("avg_final_net_worth", 0),
                "action_distribution": result.get("action_distribution", {}),
            }
        
        # Process demo episode
        if demo_data:
            dashboard_data["demo_episode"] = {
                "summary": demo_data.get("summary", {}),
                "net_worth_history": demo_data.get("net_worth_history", []),
                "regime_history": demo_data.get("regime_history", []),
                "log": demo_data.get("log", []),
            }
        
        path = os.path.join(self.output_dir, "dashboard_data.json")
        with open(path, "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"📊 Dashboard data saved to {path}")
        return path
