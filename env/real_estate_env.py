"""
Real Estate Investment Environment — Core Gymnasium Env.

A multi-property real estate investment environment where an RL agent
manages a portfolio of up to 5 properties across 3 neighborhoods,
navigating market cycles (boom/stable/recession) to maximize
risk-adjusted returns.

API: step() / reset() / render()
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .config import EnvConfig, Regime, PropertyType, Neighborhood
from .market_engine import MarketEngine
from .property_manager import PropertyManager


class RealEstateEnv(gym.Env):
    """
    Real Estate Investment RL Environment.
    
    Observation Space (69 dim):
        - Cash (normalized)                    : 1
        - Net worth (normalized)               : 1
        - Market regime one-hot                 : 3
        - Interest rate (normalized)            : 1
        - Demand index                          : 1
        - Inflation (normalized)               : 1
        - Seasonal sin/cos                      : 2
        - Time remaining (normalized)           : 1
        - Neighborhood qualities               : 3
        - Property slots (5 × 11)              : 55 → total = 69
    
    Action Space (MultiDiscrete):
        5 slots × 5 actions = MultiDiscrete([5, 5, 5, 5, 5])
        Actions per slot: 0=Hold, 1=Buy, 2=Sell, 3=Raise Rent, 4=Lower Rent
    
    Reward:
        Weighted combination of step return and rolling Sharpe ratio,
        with penalties for illegal actions, foreclosure, and high vacancy.
    """
    
    metadata = {"render_modes": ["human", "json"], "render_fps": 1}
    
    # Action constants
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_RAISE_RENT = 3
    ACTION_LOWER_RENT = 4
    ACTION_NAMES = {0: "Hold", 1: "Buy", 2: "Sell", 3: "Raise Rent", 4: "Lower Rent"}
    
    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Sub-systems
        self.market = MarketEngine(self.config)
        self.portfolio = PropertyManager(self.config)
        
        # ── Define spaces ──
        # Observation: 14 global features + 5 slots × 11 features = 69
        obs_dim = 14 + self.config.max_properties * 11
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: one discrete choice per property slot
        self.action_space = spaces.MultiDiscrete(
            [5] * self.config.max_properties
        )
        
        # ── Episode state ──
        self.cash = 0.0
        self.step_count = 0
        self.net_worth_history = []
        self.return_history = []
        self.episode_log = []
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset subsystems
        self.market.reset(seed=seed)
        self.portfolio.reset()
        
        # Reset episode state
        self.cash = self.config.initial_cash
        self.step_count = 0
        self.net_worth_history = [self.cash]
        self.return_history = []
        self.episode_log = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one month of decisions.
        
        Args:
            action: Array of 5 actions, one per property slot
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        prev_net_worth = self._calculate_net_worth()
        
        step_info = {
            "month": self.step_count,
            "actions": [],
            "penalties": 0.0,
            "cash_before": self.cash,
            "regime": Regime.NAMES[self.market.current_regime],
        }
        
        # ── 1. Process agent actions ──
        penalty = 0.0
        for slot_id in range(self.config.max_properties):
            act = int(action[slot_id])
            act_result = self._execute_action(slot_id, act)
            step_info["actions"].append(act_result)
            penalty += act_result.get("penalty", 0.0)
        
        # ── 2. Advance market ──
        market_state = self.market.step()
        
        # ── 3. Process monthly portfolio updates ──
        portfolio_update = self.portfolio.monthly_update({
            "price_fn": self.market.get_current_price,
            "rent_fn": self.market.get_market_rent,
            "demand_index": self.market.demand_index,
        })
        
        # Apply cash flow
        self.cash += portfolio_update["net_cash_flow"]
        
        # ── 4. Check for foreclosure ──
        if self.cash < -500_000:  # Allow small negative (overdraft up to ₹5L)
            penalty += self.config.foreclosure_penalty
            # Force sell worst property to cover
            self._emergency_sell()
        
        # ── 5. Calculate reward ──
        current_net_worth = self._calculate_net_worth()
        
        # Step return
        if prev_net_worth > 0:
            step_return = (current_net_worth - prev_net_worth) / prev_net_worth
        else:
            step_return = 0.0
        
        self.net_worth_history.append(current_net_worth)
        self.return_history.append(step_return)
        
        # Rolling Sharpe ratio (risk-adjusted return)
        sharpe = self._calculate_rolling_sharpe()
        
        # Vacancy penalty
        vacancy_penalty = 0.0
        for prop in self.portfolio.properties.values():
            if prop is not None and prop.occupancy < 0.5:
                vacancy_penalty += self.config.vacancy_penalty_rate
        
        # Final reward
        reward = (
            self.config.return_weight * step_return
            + self.config.sharpe_weight * sharpe * 0.01  # Scale down
            - penalty
            - vacancy_penalty
        )
        
        # ── 6. Termination / Truncation ──
        terminated = current_net_worth <= 0  # Bankrupt
        truncated = self.step_count >= self.config.episode_length
        
        # ── 7. Build info ──
        step_info.update({
            "cash_after": self.cash,
            "net_worth": current_net_worth,
            "step_return": step_return,
            "sharpe": sharpe,
            "reward": reward,
            "portfolio": self.portfolio.get_summary(),
            "market": {
                "regime": Regime.NAMES[self.market.current_regime],
                "interest_rate": round(self.market.interest_rate, 4),
                "demand": round(self.market.demand_index, 3),
                "inflation": round(self.market.inflation, 4),
            },
            "portfolio_cash_flow": portfolio_update,
        })
        
        self.episode_log.append(step_info)
        
        obs = self._get_observation()
        info = self._get_info()
        info["step_detail"] = step_info
        
        if terminated or truncated:
            info["episode_summary"] = self._get_episode_summary()
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "json":
            return self._get_info()
    
    def state(self) -> Dict[str, Any]:
        """
        Return the full current environment state as a dictionary.
        
        Part of the standard step() / reset() / state() API.
        Provides a human-readable snapshot of:
          - Portfolio (cash, net worth, properties)
          - Market conditions (regime, rates, demand, inflation)
          - Episode progress
        """
        return {
            "month": self.step_count,
            "cash": round(self.cash),
            "net_worth": round(self._calculate_net_worth()),
            "regime": Regime.NAMES[self.market.current_regime],
            "interest_rate": round(self.market.interest_rate, 4),
            "demand_index": round(self.market.demand_index, 3),
            "inflation": round(self.market.inflation, 4),
            "num_properties": self.portfolio.get_owned_count(),
            "portfolio": self.portfolio.get_summary(),
            "market": self.market.get_state(),
            "episode_progress": round(self.step_count / self.config.episode_length, 3),
        }
    
    # ─── Private: Action Execution ────────────────────────────────
    
    def _execute_action(self, slot_id: int, action: int) -> Dict:
        """Execute a single action on a property slot."""
        result = {
            "slot": slot_id,
            "action": self.ACTION_NAMES[action],
            "success": False,
            "message": "",
            "penalty": 0.0,
        }
        
        if action == self.ACTION_HOLD:
            result["success"] = True
            result["message"] = "Held"
            return result
        
        if action == self.ACTION_BUY:
            if self.portfolio.properties[slot_id] is not None:
                result["penalty"] = self.config.illegal_action_penalty
                result["message"] = "Slot occupied, can't buy"
                return result
            
            # Choose property type and neighborhood randomly weighted by market
            ptype = self.np_random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
            nhood = self.np_random.choice([0, 1, 2], p=[0.25, 0.4, 0.35])
            
            market_price = self.market.get_current_price(ptype, nhood)
            market_rent = self.market.get_market_rent(ptype, nhood)
            
            success, cash_spent, msg = self.portfolio.buy_property(
                slot_id, ptype, nhood, market_price, market_rent,
                self.market.interest_rate, self.cash
            )
            
            if success:
                self.cash -= cash_spent
                result["success"] = True
            else:
                result["penalty"] = self.config.illegal_action_penalty * 0.5
            
            result["message"] = msg
            return result
        
        if action == self.ACTION_SELL:
            if self.portfolio.properties[slot_id] is None:
                result["penalty"] = self.config.illegal_action_penalty
                result["message"] = "No property to sell"
                return result
            
            success, cash_received, msg = self.portfolio.sell_property(slot_id)
            if success:
                self.cash += cash_received
                result["success"] = True
            result["message"] = msg
            return result
        
        if action == self.ACTION_RAISE_RENT:
            if self.portfolio.properties[slot_id] is None:
                result["penalty"] = self.config.illegal_action_penalty
                result["message"] = "No property to adjust"
                return result
            
            success, msg = self.portfolio.adjust_rent(slot_id, "raise")
            result["success"] = success
            result["message"] = msg
            return result
        
        if action == self.ACTION_LOWER_RENT:
            if self.portfolio.properties[slot_id] is None:
                result["penalty"] = self.config.illegal_action_penalty
                result["message"] = "No property to adjust"
                return result
            
            success, msg = self.portfolio.adjust_rent(slot_id, "lower")
            result["success"] = success
            result["message"] = msg
            return result
        
        return result
    
    def _emergency_sell(self):
        """Force sell the lowest-equity property when cash is critically low."""
        owned = [(sid, p) for sid, p in self.portfolio.properties.items() if p is not None]
        if not owned:
            return
        
        # Sell the one with lowest equity
        owned.sort(key=lambda x: x[1].equity)
        slot_id = owned[0][0]
        success, cash_received, _ = self.portfolio.sell_property(slot_id)
        if success:
            self.cash += cash_received
    
    # ─── Private: Observation ─────────────────────────────────────
    
    def _get_observation(self) -> np.ndarray:
        """Build the full observation vector (normalized [0,1])."""
        cfg = self.config
        
        # Global features
        cash_norm = np.clip(self.cash / cfg.max_cash, 0, 1)
        net_worth = self._calculate_net_worth()
        nw_norm = np.clip(net_worth / cfg.max_net_worth, 0, 1)
        
        # Regime one-hot
        regime_oh = [0.0, 0.0, 0.0]
        regime_oh[self.market.current_regime] = 1.0
        
        # Market features
        rate_norm = np.clip(
            (self.market.interest_rate - cfg.rate_min) / (cfg.rate_max - cfg.rate_min), 0, 1
        )
        demand = np.clip(self.market.demand_index, 0, 1)
        inflation_norm = np.clip((self.market.inflation + 0.02) / 0.12, 0, 1)
        
        # Seasonal encoding
        month_sin = (np.sin(2 * np.pi * self.step_count / 12) + 1) / 2
        month_cos = (np.cos(2 * np.pi * self.step_count / 12) + 1) / 2
        
        # Time remaining
        time_remaining = 1.0 - self.step_count / cfg.episode_length
        
        # Neighborhood qualities (normalized)
        nq = [np.clip(self.market.neighborhood_quality[i] / 2.0, 0, 1) for i in range(3)]
        
        global_obs = np.array(
            [cash_norm, nw_norm] + regime_oh + [rate_norm, demand, inflation_norm,
            month_sin, month_cos, time_remaining] + nq,
            dtype=np.float32
        )
        
        # Property slot observations
        property_obs = self.portfolio.get_observation_vector()
        
        return np.concatenate([global_obs, property_obs])
    
    def _calculate_net_worth(self) -> float:
        """Total net worth = cash + property equity."""
        return self.cash + self.portfolio.get_total_equity()
    
    def _calculate_rolling_sharpe(self, window: int = 12) -> float:
        """Rolling Sharpe ratio over last `window` months."""
        if len(self.return_history) < 2:
            return 0.0
        
        returns = np.array(self.return_history[-window:])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-8:
            return 0.0
        
        return mean_ret / std_ret
    
    # ─── Private: Info & Rendering ────────────────────────────────
    
    def _get_info(self) -> Dict:
        """Get info dict."""
        return {
            "month": self.step_count,
            "cash": round(self.cash),
            "net_worth": round(self._calculate_net_worth()),
            "num_properties": self.portfolio.get_owned_count(),
            "regime": Regime.NAMES[self.market.current_regime],
            "interest_rate": round(self.market.interest_rate, 4),
            "demand": round(self.market.demand_index, 3),
        }
    
    def _get_episode_summary(self) -> Dict:
        """Summary statistics for the completed episode."""
        initial = self.config.initial_cash
        final = self._calculate_net_worth()
        total_return = (final - initial) / initial
        
        returns = np.array(self.return_history) if self.return_history else np.array([0])
        
        # Max drawdown
        cumulative = np.array(self.net_worth_history)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        max_drawdown = np.max(drawdown)
        
        # Annualized metrics
        years = self.step_count / 12
        if years > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = -1.0
        
        annualized_sharpe = (np.mean(returns) * 12) / (np.std(returns) * np.sqrt(12) + 1e-8)
        
        return {
            "initial_capital": initial,
            "final_net_worth": round(final),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "annualized_sharpe": round(annualized_sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "months": self.step_count,
            "total_properties_bought": self.portfolio.total_properties_bought,
            "total_properties_sold": self.portfolio.total_properties_sold,
            "transaction_fees": round(self.portfolio.total_transaction_fees_paid),
            "realized_gains": round(self.portfolio.realized_gains),
            "regime_history": [Regime.NAMES[r] for r in self.market.regime_history],
            "net_worth_history": [round(nw) for nw in self.net_worth_history],
        }
    
    def _render_human(self):
        """Print readable state to console."""
        info = self._get_info()
        regime_colors = {"BOOM": "🟢", "STABLE": "🟡", "RECESSION": "🔴"}
        regime = info["regime"]
        
        print(f"\n{'='*65}")
        print(f"  Month {info['month']:>3d}/{self.config.episode_length}"
              f"  {regime_colors.get(regime, '')} {regime}")
        print(f"{'='*65}")
        print(f"  💰 Cash:       ₹{info['cash']:>15,}")
        print(f"  📊 Net Worth:  ₹{info['net_worth']:>15,}")
        print(f"  🏠 Properties: {info['num_properties']}/{self.config.max_properties}")
        print(f"  📈 Interest:   {info['interest_rate']:.2%}")
        print(f"  📉 Demand:     {info['demand']:.1%}")
        
        for sid, prop in self.portfolio.properties.items():
            if prop is not None:
                cf_emoji = "✅" if prop.monthly_cash_flow > 0 else "❌"
                print(f"   Slot {sid}: {PropertyType.NAMES[prop.property_type]}"
                      f" in {Neighborhood.NAMES[prop.neighborhood]}"
                      f" | Val: ₹{prop.current_value:,.0f}"
                      f" | Rent: ₹{prop.rent:,.0f}"
                      f" | Occ: {prop.occupancy:.0%}"
                      f" | CF: ₹{prop.monthly_cash_flow:,.0f} {cf_emoji}")
        print(f"{'='*65}")
    
    def get_episode_log(self) -> List[Dict]:
        """Get full episode log for dashboard visualization."""
        return self.episode_log
