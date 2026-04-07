"""
Property Manager — Manages the agent's real estate portfolio.

Handles:
- Property buying/selling with transaction costs
- Mortgage tracking (principal + interest payments)
- Rent collection with occupancy modeling
- Maintenance costs that increase with property age
- Portfolio valuation and cash flow
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .config import EnvConfig, PropertyType, Neighborhood


@dataclass
class Property:
    """A single property in the portfolio."""
    slot_id: int
    property_type: int          # PropertyType enum
    neighborhood: int           # Neighborhood enum
    purchase_price: float       # Original purchase price
    current_value: float        # Current market value
    rent: float                 # Monthly rent being charged
    market_rent: float          # Current fair market rent
    occupancy: float            # Occupancy rate [0, 1]
    mortgage_balance: float     # Remaining mortgage
    monthly_payment: float      # Fixed monthly mortgage payment
    mortgage_rate: float        # Locked-in mortgage rate
    tax_rate: float             # Annual property tax rate from config
    age_months: int             # Months since purchase
    
    @property
    def equity(self) -> float:
        return self.current_value - self.mortgage_balance
    
    @property
    def monthly_maintenance(self) -> float:
        maint_rate = PropertyType.PROFILES[self.property_type]["maint_rate"]
        # Maintenance increases with age (1% per year extra)
        age_factor = 1.0 + (self.age_months / 12) * 0.01
        return self.current_value * maint_rate / 12 * age_factor
    
    @property
    def monthly_tax(self) -> float:
        return self.current_value * self.tax_rate / 12
    
    @property
    def monthly_cash_flow(self) -> float:
        income = self.rent * self.occupancy
        expenses = self.monthly_payment + self.monthly_maintenance + self.monthly_tax
        return income - expenses
    
    def to_dict(self) -> Dict:
        return {
            "slot_id": self.slot_id,
            "type": PropertyType.NAMES[self.property_type],
            "neighborhood": Neighborhood.NAMES[self.neighborhood],
            "purchase_price": round(self.purchase_price),
            "current_value": round(self.current_value),
            "rent": round(self.rent),
            "market_rent": round(self.market_rent),
            "occupancy": round(self.occupancy, 2),
            "mortgage_balance": round(self.mortgage_balance),
            "equity": round(self.equity),
            "monthly_cash_flow": round(self.monthly_cash_flow),
            "age_months": self.age_months,
        }


class PropertyManager:
    """Manages the agent's property portfolio."""
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.properties: Dict[int, Optional[Property]] = {}
        self.reset()
    
    def reset(self):
        """Clear all properties."""
        self.properties = {i: None for i in range(self.config.max_properties)}
        self.total_properties_bought = 0
        self.total_properties_sold = 0
        self.total_transaction_fees_paid = 0.0
        self.realized_gains = 0.0
    
    def get_owned_count(self) -> int:
        """Number of properties currently owned."""
        return sum(1 for p in self.properties.values() if p is not None)
    
    def get_empty_slot(self) -> Optional[int]:
        """Get first empty slot, or None."""
        for slot_id, prop in self.properties.items():
            if prop is None:
                return slot_id
        return None
    
    def buy_property(
        self,
        slot_id: int,
        property_type: int,
        neighborhood: int,
        market_price: float,
        market_rent: float,
        interest_rate: float,
        cash: float
    ) -> Tuple[bool, float, str]:
        """
        Buy a property. Returns (success, cash_spent, message).
        
        The agent pays the down payment + transaction costs.
        Remaining is covered by mortgage.
        """
        if self.properties[slot_id] is not None:
            return False, 0.0, "Slot occupied"
        
        # Calculate costs
        transaction_cost = market_price * self.config.transaction_cost_rate
        down_payment = market_price * (1 - self.config.mortgage_ltv)
        total_cash_needed = down_payment + transaction_cost
        
        if cash < total_cash_needed:
            return False, 0.0, f"Need \u20b9{total_cash_needed:,.0f}, have \u20b9{cash:,.0f}"
        
        # Mortgage setup
        mortgage_amount = market_price * self.config.mortgage_ltv
        monthly_rate = interest_rate / 12
        n_payments = self.config.mortgage_term_months
        
        if monthly_rate > 0:
            monthly_payment = mortgage_amount * (
                monthly_rate * (1 + monthly_rate) ** n_payments
            ) / ((1 + monthly_rate) ** n_payments - 1)
        else:
            monthly_payment = mortgage_amount / n_payments
        
        # Create property
        self.properties[slot_id] = Property(
            slot_id=slot_id,
            property_type=property_type,
            neighborhood=neighborhood,
            purchase_price=market_price,
            current_value=market_price,
            rent=market_rent,  # Start at market rent
            market_rent=market_rent,
            occupancy=self.config.base_occupancy,
            mortgage_balance=mortgage_amount,
            monthly_payment=monthly_payment,
            mortgage_rate=interest_rate,
            tax_rate=self.config.property_tax_rate,
            age_months=0,
        )
        
        self.total_properties_bought += 1
        self.total_transaction_fees_paid += transaction_cost
        
        return True, total_cash_needed, f"Bought {PropertyType.NAMES[property_type]} in {Neighborhood.NAMES[neighborhood]}"
    
    def sell_property(self, slot_id: int) -> Tuple[bool, float, str]:
        """
        Sell a property. Returns (success, cash_received, message).
        """
        prop = self.properties[slot_id]
        if prop is None:
            return False, 0.0, "No property in slot"
        
        sale_price = prop.current_value
        transaction_cost = sale_price * self.config.transaction_cost_rate
        mortgage_payoff = prop.mortgage_balance
        
        net_proceeds = sale_price - transaction_cost - mortgage_payoff
        
        # Track realized gains
        total_cost_basis = prop.purchase_price
        self.realized_gains += (sale_price - total_cost_basis)
        
        self.properties[slot_id] = None
        self.total_properties_sold += 1
        self.total_transaction_fees_paid += transaction_cost
        
        return True, net_proceeds, f"Sold for \u20b9{sale_price:,.0f}, net \u20b9{net_proceeds:,.0f}"
    
    def adjust_rent(self, slot_id: int, direction: str) -> Tuple[bool, str]:
        """
        Adjust rent up or down. Returns (success, message).
        direction: 'raise' or 'lower'
        """
        prop = self.properties[slot_id]
        if prop is None:
            return False, "No property in slot"
        
        pct = self.config.rent_adjustment_pct
        max_rent = prop.market_rent * self.config.max_rent_multiplier
        min_rent = prop.market_rent * self.config.min_rent_multiplier
        
        if direction == "raise":
            new_rent = min(prop.rent * (1 + pct), max_rent)
        else:
            new_rent = max(prop.rent * (1 - pct), min_rent)
        
        old_rent = prop.rent
        prop.rent = new_rent
        
        return True, f"Rent \u20b9{old_rent:,.0f} \u2192 \u20b9{new_rent:,.0f}"
    
    def monthly_update(self, market_state: Dict) -> Dict:
        """
        Process one month for all properties.
        Returns cash flow summary.
        """
        total_income = 0.0
        total_expenses = 0.0
        total_mortgage_principal = 0.0
        details = []
        
        for slot_id, prop in self.properties.items():
            if prop is None:
                continue
            
            prop.age_months += 1
            
            # Update property value from market
            prop.current_value = market_state["price_fn"](prop.property_type, prop.neighborhood)
            prop.market_rent = market_state["rent_fn"](prop.property_type, prop.neighborhood)
            
            # Update occupancy based on rent-to-market ratio
            rent_ratio = prop.rent / max(prop.market_rent, 1.0)
            # Occupancy drops as rent exceeds market rate
            occupancy_target = self.config.base_occupancy - self.config.occupancy_sensitivity * max(0, rent_ratio - 1.0)
            # Also affected by demand
            demand_boost = (market_state["demand_index"] - 0.5) * 0.1
            occupancy_target += demand_boost
            occupancy_target = np.clip(occupancy_target, 0.0, 1.0)
            # Smooth transition
            prop.occupancy = 0.7 * prop.occupancy + 0.3 * occupancy_target
            
            # Income
            rental_income = prop.rent * prop.occupancy
            total_income += rental_income
            
            # Expenses
            maintenance = prop.monthly_maintenance
            tax = prop.monthly_tax
            mortgage_pmt = prop.monthly_payment
            total_expenses += maintenance + tax + mortgage_pmt
            
            # Mortgage principal reduction
            if prop.mortgage_balance > 0:
                interest_portion = prop.mortgage_balance * (prop.mortgage_rate / 12)
                principal_portion = min(mortgage_pmt - interest_portion, prop.mortgage_balance)
                principal_portion = max(0, principal_portion)
                prop.mortgage_balance = max(0, prop.mortgage_balance - principal_portion)
                total_mortgage_principal += principal_portion
            
            details.append({
                "slot": slot_id,
                "income": rental_income,
                "expenses": maintenance + tax + mortgage_pmt,
                "cash_flow": rental_income - maintenance - tax - mortgage_pmt,
                "value": prop.current_value,
                "equity": prop.equity,
            })
        
        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_cash_flow": total_income - total_expenses,
            "principal_paid": total_mortgage_principal,
            "details": details,
        }
    
    def get_portfolio_value(self) -> float:
        """Total value of all properties."""
        return sum(p.current_value for p in self.properties.values() if p is not None)
    
    def get_total_equity(self) -> float:
        """Total equity across all properties."""
        return sum(p.equity for p in self.properties.values() if p is not None)
    
    def get_total_mortgage(self) -> float:
        """Total outstanding mortgage."""
        return sum(p.mortgage_balance for p in self.properties.values() if p is not None)
    
    def get_observation_vector(self) -> np.ndarray:
        """
        Get normalized observation vector for all property slots.
        Each slot: [occupied, value_norm, rent_norm, occupancy, mortgage_norm, neighborhood_onehot(3), type_onehot(3)]
        = 11 features per slot
        """
        cfg = self.config
        obs = []
        
        for slot_id in range(cfg.max_properties):
            prop = self.properties[slot_id]
            if prop is None:
                obs.extend([0.0] * 11)  # Empty slot
            else:
                # Scalar features (normalized)
                occupied = 1.0
                value_norm = np.clip(prop.current_value / cfg.max_property_value, 0, 1)
                rent_norm = np.clip(prop.rent / cfg.max_rent, 0, 1)
                occupancy = prop.occupancy
                mortgage_norm = np.clip(prop.mortgage_balance / cfg.max_property_value, 0, 1)
                
                # One-hot neighborhood
                nhood_oh = [0.0, 0.0, 0.0]
                nhood_oh[prop.neighborhood] = 1.0
                
                # One-hot property type
                type_oh = [0.0, 0.0, 0.0]
                type_oh[prop.property_type] = 1.0
                
                obs.extend([occupied, value_norm, rent_norm, occupancy, mortgage_norm] + nhood_oh + type_oh)
        
        return np.array(obs, dtype=np.float32)
    
    def get_summary(self) -> Dict:
        """Human-readable portfolio summary."""
        props = [p.to_dict() for p in self.properties.values() if p is not None]
        return {
            "num_properties": self.get_owned_count(),
            "portfolio_value": round(self.get_portfolio_value()),
            "total_equity": round(self.get_total_equity()),
            "total_mortgage": round(self.get_total_mortgage()),
            "total_bought": self.total_properties_bought,
            "total_sold": self.total_properties_sold,
            "transaction_fees_paid": round(self.total_transaction_fees_paid),
            "realized_gains": round(self.realized_gains),
            "properties": props,
        }
