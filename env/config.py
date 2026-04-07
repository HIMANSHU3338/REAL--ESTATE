"""
Configuration & hyperparameters for the Real Estate RL Environment.
Indian Real Estate Market (₹) — All prices in Indian Rupees.
All tunable constants live here — no magic numbers elsewhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# ─── Market Regimes ───────────────────────────────────────────────
class Regime:
    BOOM = 0
    STABLE = 1
    RECESSION = 2
    NAMES = {0: "BOOM", 1: "STABLE", 2: "RECESSION"}


# ─── Property Types ──────────────────────────────────────────────
class PropertyType:
    APARTMENT = 0
    HOUSE = 1
    COMMERCIAL = 2
    NAMES = {0: "Apartment", 1: "House", 2: "Commercial"}
    
    # Base characteristics in ₹: (base_price, base_rent/month, maintenance_rate)
    # Apartment: ₹50 Lakh, House: ₹1.2 Crore, Commercial: ₹2.5 Crore
    PROFILES = {
        0: {"base_price": 5_000_000, "base_rent": 25_000, "maint_rate": 0.005},
        1: {"base_price": 12_000_000, "base_rent": 50_000, "maint_rate": 0.008},
        2: {"base_price": 25_000_000, "base_rent": 150_000, "maint_rate": 0.006},
    }


# ─── Neighborhoods ───────────────────────────────────────────────
class Neighborhood:
    SOUTH_MUMBAI = 0
    BANDRA = 1
    NAVI_MUMBAI = 2
    NAMES = {0: "South Mumbai", 1: "Bandra", 2: "Navi Mumbai"}
    
    # Quality multiplier: affects price and rent
    QUALITY = {0: 1.3, 1: 1.0, 2: 0.75}
    # Gentrification drift per month (can be positive or negative)
    DRIFT = {0: 0.001, 1: 0.0005, 2: 0.002}


@dataclass
class EnvConfig:
    """Master configuration for the environment."""
    
    # ── Episode ──
    episode_length: int = 120  # 10 years of monthly decisions
    initial_cash: float = 20_000_000.0  # ₹2 Crore starting capital
    
    # ── Portfolio ──
    max_properties: int = 5
    
    # ── Market Regime Transition Matrix ──
    # From → To: [BOOM, STABLE, RECESSION]
    regime_transitions: np.ndarray = field(default_factory=lambda: np.array([
        [0.85, 0.12, 0.03],   # BOOM → ...
        [0.08, 0.82, 0.10],   # STABLE → ...
        [0.05, 0.25, 0.70],   # RECESSION → ...
    ]))
    
    # ── Market Parameters per Regime ──
    # (price_drift_monthly, demand_mean, demand_std, rate_drift)
    regime_params: Dict = field(default_factory=lambda: {
        Regime.BOOM:      {"price_drift": 0.005, "demand_mean": 0.8, "demand_std": 0.1, "rate_drift": 0.001},
        Regime.STABLE:    {"price_drift": 0.001, "demand_mean": 0.5, "demand_std": 0.15, "rate_drift": 0.0},
        Regime.RECESSION: {"price_drift": -0.003, "demand_mean": 0.25, "demand_std": 0.2, "rate_drift": -0.001},
    })
    
    # ── Interest Rate (RBI repo-rate inspired, Vasicek-like) ──
    initial_interest_rate: float = 0.065   # 6.5% annual (typical Indian home loan)
    rate_mean_reversion: float = 0.1
    rate_long_term_mean: float = 0.065
    rate_volatility: float = 0.005
    rate_min: float = 0.04
    rate_max: float = 0.12
    
    # ── Financial (Indian market) ──
    transaction_cost_rate: float = 0.07   # 7% stamp duty + registration in India
    property_tax_rate: float = 0.005      # 0.5% annual (Indian municipal tax)
    mortgage_ltv: float = 0.75            # 75% loan-to-value (typical Indian bank)
    mortgage_term_months: int = 240       # 20 years (common in India)
    rent_adjustment_pct: float = 0.05     # 5% raise/lower per action
    max_rent_multiplier: float = 1.8      # Max rent vs base market rent
    min_rent_multiplier: float = 0.5      # Min rent vs base market rent
    
    # ── Occupancy Model ──
    base_occupancy: float = 0.92          # 92% when rent = market rate
    occupancy_sensitivity: float = 0.3    # How much rent-to-market ratio affects occupancy
    
    # ── Reward Shaping ──
    sharpe_weight: float = 0.4
    return_weight: float = 0.6
    illegal_action_penalty: float = -0.05
    foreclosure_penalty: float = -0.5
    vacancy_penalty_rate: float = 0.01
    
    # ── Normalization Bounds (₹) ──
    max_cash: float = 200_000_000.0       # ₹20 Crore
    max_property_value: float = 100_000_000.0  # ₹10 Crore
    max_rent: float = 500_000.0           # ₹5 Lakh/month
    max_net_worth: float = 500_000_000.0  # ₹50 Crore
    
    # ── Seasonality ──
    seasonal_amplitude: float = 0.05  # 5% seasonal demand swing
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.episode_length > 0
        assert self.initial_cash > 0
        assert self.max_properties > 0
        assert 0 < self.mortgage_ltv < 1
        row_sums = self.regime_transitions.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Transition rows must sum to 1, got {row_sums}"
