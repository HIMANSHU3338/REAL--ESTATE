"""
Baseline Agents for benchmarking the RL agent.

Three deterministic agents:
1. RandomAgent — random actions (lower bound)
2. BuyAndHoldAgent — buys when possible, holds forever
3. RuleBasedAgent — smart heuristics: buy in recession, sell in boom
"""

import numpy as np
from typing import Dict


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, name: str, num_slots: int = 5):
        self.name = name
        self.num_slots = num_slots
    
    def predict(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """Predict action given observation. Returns action array."""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.name}Agent"


class RandomAgent(BaseAgent):
    """Takes completely random actions. Lower bound baseline."""
    
    def __init__(self, num_slots: int = 5, seed: int = 42):
        super().__init__("Random", num_slots)
        self.rng = np.random.default_rng(seed)
    
    def predict(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        return self.rng.integers(0, 5, size=self.num_slots)


class BuyAndHoldAgent(BaseAgent):
    """
    Buys whenever there's an empty slot and enough cash.
    Never sells. Doesn't adjust rent.
    Classic passive investor strategy.
    """
    
    def __init__(self, num_slots: int = 5):
        super().__init__("BuyAndHold", num_slots)
    
    def predict(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        actions = np.zeros(self.num_slots, dtype=np.int64)
        
        for slot in range(self.num_slots):
            # Check if slot is empty (first feature of each slot block is 'occupied')
            # Global features = 14, each slot = 11 features
            slot_offset = 14 + slot * 11
            if slot_offset < len(obs):
                occupied = obs[slot_offset]
                if occupied < 0.5:  # Empty slot
                    actions[slot] = 1  # Buy
                else:
                    actions[slot] = 0  # Hold
        
        return actions


class RuleBasedAgent(BaseAgent):
    """
    Smart heuristic agent:
    - Buys in RECESSION (when prices are low)
    - Sells in BOOM (when prices are high) — but only if profitable
    - Adjusts rent based on occupancy
    - Holds in STABLE market
    
    This represents a well-informed human investor.
    The RL agent should aim to beat this.
    """
    
    def __init__(self, num_slots: int = 5):
        super().__init__("RuleBased", num_slots)
    
    def predict(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        actions = np.zeros(self.num_slots, dtype=np.int64)
        
        # Decode regime from one-hot (indices 2, 3, 4 in observation)
        regime_oh = obs[2:5]
        regime = np.argmax(regime_oh)  # 0=BOOM, 1=STABLE, 2=RECESSION
        
        # Decode demand (index 6)
        demand = obs[6] if len(obs) > 6 else 0.5
        
        # Cash level (index 0)
        cash_norm = obs[0]
        
        for slot in range(self.num_slots):
            slot_offset = 14 + slot * 11
            if slot_offset >= len(obs):
                continue
            
            occupied = obs[slot_offset]
            
            if occupied < 0.5:
                # Empty slot — buy strategy
                if regime == 2 and cash_norm > 0.02:  # RECESSION — buy aggressively
                    actions[slot] = 1  # Buy (prices are low)
                elif regime == 1 and cash_norm > 0.04:  # STABLE with some cash
                    actions[slot] = 1  # Buy cautiously
                elif regime == 0 and cash_norm > 0.08:  # BOOM — buy if plenty of cash
                    actions[slot] = 1
                else:
                    actions[slot] = 0  # Hold (wait)
            else:
                # Owned property
                occupancy = obs[slot_offset + 3] if slot_offset + 3 < len(obs) else 0.9
                
                if regime == 0:  # BOOM
                    # Consider selling if we own enough
                    owned_count = sum(
                        1 for s in range(self.num_slots)
                        if 14 + s * 11 < len(obs) and obs[14 + s * 11] > 0.5
                    )
                    if owned_count > 2 and cash_norm < 0.1:
                        actions[slot] = 2  # Sell (take profits)
                    elif demand > 0.6:
                        actions[slot] = 3  # Raise rent (high demand)
                    else:
                        actions[slot] = 0  # Hold
                
                elif regime == 2:  # RECESSION
                    if occupancy < 0.7:
                        actions[slot] = 4  # Lower rent (attract tenants)
                    else:
                        actions[slot] = 0  # Hold
                
                else:  # STABLE
                    if occupancy > 0.9 and demand > 0.5:
                        actions[slot] = 3  # Raise rent gently
                    elif occupancy < 0.75:
                        actions[slot] = 4  # Lower rent
                    else:
                        actions[slot] = 0  # Hold
        
        return actions
