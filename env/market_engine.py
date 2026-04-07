"""
Market Engine — Simulates realistic real estate macro conditions.

Features:
- Regime switching (Boom / Stable / Recession) via Markov chain
- Mean-reverting interest rates (Vasicek model)
- Demand dynamics with seasonal effects
- Property price evolution driven by fundamentals
- Neighborhood quality drift (gentrification / decline)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .config import EnvConfig, Regime, Neighborhood, PropertyType


class MarketEngine:
    """Drives all macro-level market simulation."""
    
    def __init__(self, config: EnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset market to initial conditions."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Current state
        self.current_regime = Regime.STABLE
        self.interest_rate = self.config.initial_interest_rate
        self.demand_index = 0.5  # Normalized [0, 1]
        self.inflation = 0.03   # 3% annual baseline
        self.month = 0
        
        # Neighborhood quality multipliers (can drift over time)
        self.neighborhood_quality = {
            k: v for k, v in Neighborhood.QUALITY.items()
        }
        
        # Price index per property type per neighborhood (base = 1.0)
        self.price_indices = {}
        for ptype in [PropertyType.APARTMENT, PropertyType.HOUSE, PropertyType.COMMERCIAL]:
            for nhood in [Neighborhood.SOUTH_MUMBAI, Neighborhood.BANDRA, Neighborhood.NAVI_MUMBAI]:
                self.price_indices[(ptype, nhood)] = 1.0
        
        # History for analytics
        self.regime_history = [self.current_regime]
        self.rate_history = [self.interest_rate]
        self.demand_history = [self.demand_index]
        self.price_index_history = [self._avg_price_index()]
        
        return self.get_state()
    
    def step(self) -> Dict:
        """Advance market by one month. Returns new market state."""
        self.month += 1
        
        # 1. Regime transition
        self._transition_regime()
        
        # 2. Interest rate update (Vasicek)
        self._update_interest_rate()
        
        # 3. Demand update
        self._update_demand()
        
        # 4. Inflation update
        self._update_inflation()
        
        # 5. Neighborhood quality drift
        self._update_neighborhoods()
        
        # 6. Price indices update
        self._update_price_indices()
        
        # Track history
        self.regime_history.append(self.current_regime)
        self.rate_history.append(self.interest_rate)
        self.demand_history.append(self.demand_index)
        self.price_index_history.append(self._avg_price_index())
        
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Return current market state as a dictionary."""
        return {
            "regime": self.current_regime,
            "interest_rate": self.interest_rate,
            "demand_index": self.demand_index,
            "inflation": self.inflation,
            "month": self.month,
            "neighborhood_quality": dict(self.neighborhood_quality),
            "price_indices": dict(self.price_indices),
            "seasonal_factor": self._seasonal_factor(),
        }
    
    def get_current_price(self, property_type: int, neighborhood: int) -> float:
        """Get current market price for a property type in a neighborhood."""
        base = PropertyType.PROFILES[property_type]["base_price"]
        quality = self.neighborhood_quality[neighborhood]
        price_idx = self.price_indices[(property_type, neighborhood)]
        return base * quality * price_idx
    
    def get_market_rent(self, property_type: int, neighborhood: int) -> float:
        """Get current fair market rent for a property type in a neighborhood."""
        base_rent = PropertyType.PROFILES[property_type]["base_rent"]
        quality = self.neighborhood_quality[neighborhood]
        # Rent moves with demand and price index, but slower
        price_idx = self.price_indices[(property_type, neighborhood)]
        demand_factor = 0.8 + 0.4 * self.demand_index  # [0.8, 1.2]
        return base_rent * quality * (0.5 + 0.5 * price_idx) * demand_factor
    
    # ─── Private Methods ──────────────────────────────────────────
    
    def _transition_regime(self):
        """Markov chain regime transition."""
        probs = self.config.regime_transitions[self.current_regime]
        self.current_regime = self.rng.choice([0, 1, 2], p=probs)
    
    def _update_interest_rate(self):
        """Vasicek mean-reverting interest rate model with regime influence."""
        regime_params = self.config.regime_params[self.current_regime]
        
        kappa = self.config.rate_mean_reversion
        theta = self.config.rate_long_term_mean + regime_params["rate_drift"] * 10
        sigma = self.config.rate_volatility
        
        # Vasicek: dr = kappa * (theta - r) * dt + sigma * dW
        dt = 1.0 / 12.0  # Monthly
        dW = self.rng.normal(0, np.sqrt(dt))
        dr = kappa * (theta - self.interest_rate) * dt + sigma * dW
        
        self.interest_rate = np.clip(
            self.interest_rate + dr,
            self.config.rate_min,
            self.config.rate_max
        )
    
    def _update_demand(self):
        """Update demand index with regime influence + seasonality."""
        regime_params = self.config.regime_params[self.current_regime]
        
        # Mean-revert to regime-specific demand
        target = regime_params["demand_mean"]
        noise = self.rng.normal(0, regime_params["demand_std"]) * 0.1
        seasonal = self._seasonal_factor() * self.config.seasonal_amplitude
        
        self.demand_index = np.clip(
            0.9 * self.demand_index + 0.1 * target + noise + seasonal,
            0.0, 1.0
        )
    
    def _update_inflation(self):
        """Simple inflation model tied to regime."""
        regime_targets = {
            Regime.BOOM: 0.05,
            Regime.STABLE: 0.03,
            Regime.RECESSION: 0.01,
        }
        target = regime_targets[self.current_regime]
        noise = self.rng.normal(0, 0.005)
        self.inflation = np.clip(
            0.95 * self.inflation + 0.05 * target + noise,
            -0.02, 0.10
        )
    
    def _update_neighborhoods(self):
        """Neighborhood quality drifts over time (gentrification / decline)."""
        for nhood in [Neighborhood.SOUTH_MUMBAI, Neighborhood.BANDRA, Neighborhood.NAVI_MUMBAI]:
            drift = Neighborhood.DRIFT[nhood]
            noise = self.rng.normal(0, 0.002)
            self.neighborhood_quality[nhood] = np.clip(
                self.neighborhood_quality[nhood] + drift + noise,
                0.5, 2.0
            )
    
    def _update_price_indices(self):
        """Update property price indices based on fundamentals."""
        regime_params = self.config.regime_params[self.current_regime]
        base_drift = regime_params["price_drift"]
        
        for key in self.price_indices:
            ptype, nhood = key
            # Drift depends on regime + demand + neighborhood quality change
            demand_effect = (self.demand_index - 0.5) * 0.005
            rate_effect = -(self.interest_rate - 0.05) * 0.01  # Higher rates = lower prices
            noise = self.rng.normal(0, 0.008)
            
            change = base_drift + demand_effect + rate_effect + noise
            self.price_indices[key] = max(0.3, self.price_indices[key] * (1 + change))
    
    def _seasonal_factor(self) -> float:
        """Seasonal demand: peaks in May-June, troughs in Dec-Jan."""
        return np.sin(2 * np.pi * (self.month - 3) / 12)
    
    def _avg_price_index(self) -> float:
        """Average price index across all property types & neighborhoods."""
        return np.mean(list(self.price_indices.values()))
