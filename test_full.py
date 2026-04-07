"""Full validation test script for the Real Estate RL Environment."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig, Regime, PropertyType, Neighborhood
from env.market_engine import MarketEngine
from env.property_manager import PropertyManager
from agents.baselines import RandomAgent, BuyAndHoldAgent, RuleBasedAgent
from utils.logger import EpisodeLogger

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


print("=" * 70)
print("TEST 1: Config Validation")
print("=" * 70)
config = EnvConfig()
check("episode_length > 0", config.episode_length > 0)
check("initial_cash > 0", config.initial_cash > 0)
check("max_properties > 0", config.max_properties > 0)
check("mortgage_ltv in (0,1)", 0 < config.mortgage_ltv < 1)
row_sums = config.regime_transitions.sum(axis=1)
check("transition rows sum to 1", np.allclose(row_sums, 1.0), f"got {row_sums}")
check("Regime NAMES correct", Regime.NAMES == {0: "BOOM", 1: "STABLE", 2: "RECESSION"})
check("PropertyType PROFILES has 3 types", len(PropertyType.PROFILES) == 3)
check("Neighborhood QUALITY has 3", len(Neighborhood.QUALITY) == 3)


print("\n" + "=" * 70)
print("TEST 2: Environment Creation & Spaces")
print("=" * 70)
env = RealEstateEnv(config=config)
check("env created", env is not None)
check("observation_space shape", env.observation_space.shape == (69,), f"got {env.observation_space.shape}")
check("action_space nvec", list(env.action_space.nvec) == [5,5,5,5,5], f"got {env.action_space.nvec}")
check("obs low=0", np.all(env.observation_space.low == 0.0))
check("obs high=1", np.all(env.observation_space.high == 1.0))
check("obs dtype float32", env.observation_space.dtype == np.float32)


print("\n" + "=" * 70)
print("TEST 3: reset() API")
print("=" * 70)
obs, info = env.reset(seed=42)
check("reset returns tuple of 2", isinstance(obs, np.ndarray) and isinstance(info, dict))
check("obs shape matches space", obs.shape == env.observation_space.shape, f"got {obs.shape}")
check("obs in bounds [0,1]", np.all(obs >= 0.0) and np.all(obs <= 1.0), f"min={obs.min()}, max={obs.max()}")
check("obs dtype float32", obs.dtype == np.float32)
check("no NaN in obs", not np.any(np.isnan(obs)))
check("info has 'month'", "month" in info)
check("info has 'cash'", "cash" in info)
check("info has 'net_worth'", "net_worth" in info)
check("info has 'regime'", "regime" in info)
check("month=0 after reset", info["month"] == 0)
check("cash=initial_cash", info["cash"] == config.initial_cash)


print("\n" + "=" * 70)
print("TEST 4: step() API")
print("=" * 70)
action = env.action_space.sample()
obs2, reward, terminated, truncated, info2 = env.step(action)
check("step returns 5-tuple", True)
check("obs shape after step", obs2.shape == env.observation_space.shape)
check("obs in bounds after step", np.all(obs2 >= 0.0) and np.all(obs2 <= 1.0), f"min={obs2.min():.4f}, max={obs2.max():.4f}")
check("reward is float", isinstance(reward, float))
check("no NaN reward", not np.isnan(reward))
check("terminated is bool-like", isinstance(terminated, (bool, np.bool_)))
check("truncated is bool-like", isinstance(truncated, (bool, np.bool_)))
check("info is dict", isinstance(info2, dict))
check("info has step_detail", "step_detail" in info2)
check("month advanced to 1", info2["month"] == 1)


print("\n" + "=" * 70)
print("TEST 5: Full Episode (no crashes, no NaN)")
print("=" * 70)
env3 = RealEstateEnv(config=config)
obs, info = env3.reset(seed=99)
total_steps = 0
nan_obs = False
nan_reward = False
obs_out_of_bounds = False

while True:
    action = env3.action_space.sample()
    obs, reward, terminated, truncated, info = env3.step(action)
    total_steps += 1
    if np.any(np.isnan(obs)):
        nan_obs = True
        break
    if np.isnan(reward):
        nan_reward = True
        break
    if np.any(obs < 0.0) or np.any(obs > 1.0):
        obs_out_of_bounds = True
    if terminated or truncated:
        break

check("episode completed", terminated or truncated)
check("episode ran 120 steps", total_steps == config.episode_length or terminated, f"ran {total_steps}")
check("no NaN in observations", not nan_obs)
check("no NaN in rewards", not nan_reward)
check("all obs in [0,1]", not obs_out_of_bounds)
summary = info.get("episode_summary", {})
check("episode_summary present at end", len(summary) > 0)
check("summary has final_net_worth", "final_net_worth" in summary)
check("summary has total_return_pct", "total_return_pct" in summary)
check("summary has annualized_sharpe", "annualized_sharpe" in summary)
check("summary has max_drawdown_pct", "max_drawdown_pct" in summary)
check("summary has net_worth_history", "net_worth_history" in summary)
check("summary has regime_history", "regime_history" in summary)


print("\n" + "=" * 70)
print("TEST 6: Market Engine")
print("=" * 70)
me = MarketEngine(config, seed=42)
state = me.get_state()
check("market has regime", "regime" in state)
check("market has interest_rate", "interest_rate" in state)
check("market has demand_index", "demand_index" in state)
check("initial regime is STABLE", me.current_regime == Regime.STABLE)
check("initial rate = config", me.interest_rate == config.initial_interest_rate)

# Step market 120 times
for _ in range(120):
    me.step()
check("market survived 120 steps", True)
check("rate in bounds", config.rate_min <= me.interest_rate <= config.rate_max,
      f"rate={me.interest_rate}")
check("demand in [0,1]", 0 <= me.demand_index <= 1, f"demand={me.demand_index}")

price = me.get_current_price(PropertyType.APARTMENT, Neighborhood.BANDRA)
rent = me.get_market_rent(PropertyType.APARTMENT, Neighborhood.BANDRA)
check("price > 0", price > 0, f"price={price}")
check("rent > 0", rent > 0, f"rent={rent}")


print("\n" + "=" * 70)
print("TEST 7: Property Manager")
print("=" * 70)
pm = PropertyManager(config)
check("PM created", pm is not None)
check("0 properties initially", pm.get_owned_count() == 0)
check("empty slot available", pm.get_empty_slot() is not None)

# Buy a property
success, cash, msg = pm.buy_property(0, PropertyType.APARTMENT, Neighborhood.BANDRA,
                                     5_000_000, 25_000, 0.065, 20_000_000)
check("buy succeeds", success, msg)
check("1 property owned", pm.get_owned_count() == 1)
check("cash spent > 0", cash > 0)

# Observation vector
obs_vec = pm.get_observation_vector()
check("obs vector length = 55", len(obs_vec) == 55, f"got {len(obs_vec)}")
check("obs vector dtype float32", obs_vec.dtype == np.float32)

# Sell property
success, proceeds, msg = pm.sell_property(0)
check("sell succeeds", success, msg)
check("0 properties after sell", pm.get_owned_count() == 0)

# Buy with insufficient cash
success, _, msg = pm.buy_property(0, PropertyType.COMMERCIAL, Neighborhood.SOUTH_MUMBAI,
                                  25_000_000, 150_000, 0.065, 1_000)
check("buy fails with no cash", not success, msg)


print("\n" + "=" * 70)
print("TEST 8: Baseline Agents")
print("=" * 70)
env4 = RealEstateEnv(config=config)

for AgentClass, name in [(RandomAgent, "Random"), (BuyAndHoldAgent, "BuyAndHold"), (RuleBasedAgent, "RuleBased")]:
    if name == "Random":
        agent = AgentClass(num_slots=config.max_properties, seed=42)
    else:
        agent = AgentClass(num_slots=config.max_properties)
    
    obs, info = env4.reset(seed=42)
    total_reward = 0
    done = False
    steps = 0
    while not done:
        action = agent.predict(obs, info)
        check(f"{name} action shape", action.shape == (config.max_properties,), f"got {action.shape}")
        obs, reward, terminated, truncated, info = env4.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        if steps > 200:
            break
    
    check(f"{name} completes episode", done)
    summary = info.get("episode_summary", {})
    nw = summary.get("final_net_worth", 0)
    print(f"    {name}: NW={nw:,}, Return={summary.get('total_return_pct', 0):.1f}%, Reward={total_reward:.3f}")


print("\n" + "=" * 70)
print("TEST 9: Gymnasium check_env")
print("=" * 70)
try:
    from gymnasium.utils.env_checker import check_env
    env5 = RealEstateEnv(config=EnvConfig())
    check_env(env5)
    check("gymnasium check_env passed", True)
except Exception as e:
    check("gymnasium check_env passed", False, str(e))


print("\n" + "=" * 70)
print("TEST 10: Render modes")
print("=" * 70)
env_json = RealEstateEnv(config=EnvConfig(), render_mode="json")
obs, info = env_json.reset(seed=0)
env_json.step(env_json.action_space.sample())
render_out = env_json.render()
check("json render returns dict", isinstance(render_out, dict))
check("json render has month", "month" in render_out)

env_none = RealEstateEnv(config=EnvConfig())
check("no render_mode is None", env_none.render_mode is None)


print("\n" + "=" * 70)
print("TEST 11: Seeded Reproducibility")
print("=" * 70)
env_a = RealEstateEnv(config=EnvConfig())
obs_a, _ = env_a.reset(seed=42)
act = env_a.action_space.sample()

env_b = RealEstateEnv(config=EnvConfig())
obs_b, _ = env_b.reset(seed=42)

check("same seed -> same initial obs", np.allclose(obs_a, obs_b))


print("\n" + "=" * 70)
print("TEST 12: Episode Log")
print("=" * 70)
env6 = RealEstateEnv(config=EnvConfig())
obs, info = env6.reset(seed=0)
for _ in range(5):
    env6.step(env6.action_space.sample())
log = env6.get_episode_log()
check("episode log has 5 entries", len(log) == 5, f"got {len(log)}")
check("log entry has 'month'", "month" in log[0])
check("log entry has 'actions'", "actions" in log[0])
check("log entry has 'regime'", "regime" in log[0])


print("\n" + "=" * 70)
print("TEST 13: state() API")
print("=" * 70)
env_state = RealEstateEnv(config=EnvConfig())
obs, info = env_state.reset(seed=0)
st = env_state.state()
check("state() returns dict", isinstance(st, dict))
check("state has month", "month" in st)
check("state has cash", "cash" in st)
check("state has net_worth", "net_worth" in st)
check("state has regime", "regime" in st)
check("state has interest_rate", "interest_rate" in st)
check("state has demand_index", "demand_index" in st)
check("state has inflation", "inflation" in st)
check("state has num_properties", "num_properties" in st)
check("state has portfolio", "portfolio" in st)
check("state has market", "market" in st)
check("state has episode_progress", "episode_progress" in st)
check("state month=0 after reset", st["month"] == 0)
check("state episode_progress=0 after reset", st["episode_progress"] == 0.0)

# State after a step
env_state.step(env_state.action_space.sample())
st2 = env_state.state()
check("state month=1 after step", st2["month"] == 1)
check("state regime is valid", st2["regime"] in ["BOOM", "STABLE", "RECESSION"])


print("\n" + "=" * 70)
print("TEST 14: Logger Utility")
print("=" * 70)
logger = EpisodeLogger(output_dir="results")
check("logger created", logger is not None)


print("\n" + "=" * 70)
print(f"RESULTS: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} tests")
print("=" * 70)

if FAIL > 0:
    print("\nFAILED TESTS NEED ATTENTION!")
    sys.exit(1)
else:
    print("\nALL TESTS PASSED!")
    sys.exit(0)
