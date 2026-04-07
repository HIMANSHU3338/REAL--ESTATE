"""
Gradio App for Real Estate RL Environment
Interactive web interface for the Real Estate Investment RL environment.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig
from agents.baselines import RandomAgent, BuyAndHoldAgent, RuleBasedAgent

# Global environment instance
env = None
current_agent = None

def initialize_environment():
    """Initialize the RL environment."""
    global env
    if env is None:
        config = EnvConfig()
        env = RealEstateEnv(config=config)
    return env

def reset_environment(seed=None):
    """Reset the environment and return initial state."""
    global env, current_agent
    env = initialize_environment()

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()

    current_agent = None
    return format_observation(obs, info)

def format_observation(obs, info):
    """Format observation data for display."""
    net_worth = obs[0] * 1000000  # Scale back to rupees
    cash = obs[1] * 1000000
    step = obs[2]

    # Property values
    property_info = []
    for i in range(5):
        prop_idx = 3 + i * 13
        if obs[prop_idx] > 0:  # Property owned
            value = obs[prop_idx] * 1000000
            mortgage = obs[prop_idx + 1] * 1000000
            rent = obs[prop_idx + 2] * 1000
            property_info.append(f"Property {i+1}: ₹{value:,.0f} (Mortgage: ₹{mortgage:,.0f}, Rent: ₹{rent:,.0f})")
        else:
            property_info.append(f"Property {i+1}: Not owned")

    market_regime = ["Stable", "Boom", "Bust"][int(obs[-1])]

    result = f"""
**Step {int(step)} - Market: {market_regime}**

**Portfolio Value:** ₹{net_worth:,.0f}
**Cash:** ₹{cash:,.0f}

**Properties:**
{"\n".join(property_info)}

**Market Indicators:**
- Interest Rate: {obs[-3]:.1%}
- Demand Index: {obs[-2]:.2f}
"""

    return result

def take_action(action_str):
    """Take an action in the environment."""
    global env

    if env is None:
        return "Please reset the environment first!"

    try:
        # Parse action string like "0,1,2,3,4" into list
        action = [int(x.strip()) for x in action_str.split(',')]
        if len(action) != 5:
            return "Please provide exactly 5 actions (0-4) separated by commas!"

        obs, reward, done, truncated, info = env.step(action)

        result = format_observation(obs, info)
        result += f"\n**Reward:** {reward:.3f}"
        result += f"\n**Episode Done:** {done or truncated}"

        if done or truncated:
            result += "\n\n🎯 Episode completed! Reset to start a new episode."

        return result

    except Exception as e:
        return f"Error: {str(e)}"

def run_baseline_agent(agent_type, num_episodes=1):
    """Run a baseline agent for evaluation."""
    global env

    env = initialize_environment()

    if agent_type == "Random":
        agent = RandomAgent(env)
    elif agent_type == "Buy & Hold":
        agent = BuyAndHoldAgent(env)
    elif agent_type == "Rule-Based":
        agent = RuleBasedAgent(env)
    else:
        return "Invalid agent type!"

    results = []
    for ep in range(int(num_episodes)):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 120:  # Max 120 steps
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = done or truncated

        final_net_worth = obs[0] * 1000000
        results.append(f"Episode {ep+1}: {steps} steps, Reward={total_reward:.3f}, Net Worth=₹{final_net_worth:,.0f}")

    return "\n".join(results)

# Create Gradio interface
with gr.Blocks(title="Real Estate Investment RL", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Real Estate Investment RL Environment")
    gr.Markdown("Interactive reinforcement learning environment for real estate portfolio management.")

    with gr.Tab("Manual Control"):
        gr.Markdown("### Manual Environment Control")
        with gr.Row():
            with gr.Column():
                seed_input = gr.Number(label="Random Seed (optional)", value=None)
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
            with gr.Column():
                action_input = gr.Textbox(
                    label="Action (5 numbers 0-4, comma-separated)",
                    placeholder="0,1,2,3,4",
                    value="0,0,0,0,0"
                )
                step_btn = gr.Button("▶️ Take Action", variant="secondary")

        output_display = gr.Markdown(label="Environment State")

        reset_btn.click(
            fn=reset_environment,
            inputs=[seed_input],
            outputs=[output_display]
        )

        step_btn.click(
            fn=take_action,
            inputs=[action_input],
            outputs=[output_display]
        )

    with gr.Tab("Baseline Agents"):
        gr.Markdown("### Test Baseline Agents")
        with gr.Row():
            agent_type = gr.Dropdown(
                ["Random", "Buy & Hold", "Rule-Based"],
                label="Agent Type",
                value="Buy & Hold"
            )
            episodes_input = gr.Number(label="Number of Episodes", value=1, minimum=1, maximum=10)
            run_agent_btn = gr.Button("🚀 Run Agent", variant="primary")

        agent_output = gr.Markdown(label="Agent Results")

        run_agent_btn.click(
            fn=run_baseline_agent,
            inputs=[agent_type, episodes_input],
            outputs=[agent_output]
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Environment

        **Features:**
        - Portfolio management of up to 5 properties
        - Dynamic market regimes (Stable/Boom/Bust)
        - Mortgage financing and leverage
        - Risk-adjusted Sharpe ratio rewards
        - Three distinct neighborhoods

        **Actions per property:**
        - 0: Hold
        - 1: Buy
        - 2: Sell
        - 3: Refinance
        - 4: Develop/Renovate

        **Built with:** Gymnasium, NumPy, Matplotlib
        **License:** MIT
        """)

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")