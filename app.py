"""
Minimal Gradio app for Real Estate RL.
This app avoids audio processing and works on Hugging Face Spaces.
"""

import gradio as gr
from pathlib import Path
import sys

# Add project root to path so env imports work on Spaces
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig

env = None


def initialize_environment():
    global env
    if env is None:
        env = RealEstateEnv(config=EnvConfig())
    return env


def format_observation(obs, info):
    net_worth = obs[0] * 1_000_000
    cash = obs[1] * 1_000_000
    step = int(obs[2])
    regime = ["Stable", "Boom", "Bust"][int(obs[-1])]

    lines = [
        f"**Step {step} — Market: {regime}**",
        f"**Portfolio Value:** ₹{net_worth:,.0f}",
        f"**Cash:** ₹{cash:,.0f}",
        "",
        "**Properties:**",
    ]

    for i in range(5):
        offset = 3 + i * 13
        if obs[offset] > 0:
            value = obs[offset] * 1_000_000
            mortgage = obs[offset + 1] * 1_000_000
            rent = obs[offset + 2] * 1_000
            lines.append(
                f"Property {i+1}: ₹{value:,.0f} | Mortgage ₹{mortgage:,.0f} | Rent ₹{rent:,.0f}"
            )
        else:
            lines.append(f"Property {i+1}: Not owned")

    lines += [
        "",
        "**Market Indicators:**",
        f"- Interest rate: {obs[-3]:.1%}",
        f"- Demand index: {obs[-2]:.2f}",
    ]

    return "\n".join(lines)


def reset_environment(seed=None):
    env = initialize_environment()
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    return format_observation(obs, info)


def take_action(actions):
    env = initialize_environment()
    try:
        action = [int(x.strip()) for x in actions.split(",")]
        if len(action) != 5:
            return "Enter exactly 5 comma-separated actions (0-4)."

        obs, reward, done, truncated, info = env.step(action)
        result = format_observation(obs, info)
        result += f"\n\n**Reward:** {reward:.3f}"
        result += f"\n**Done:** {done or truncated}"
        return result
    except Exception as e:
        return f"Error: {e}"


def create_ui():
    with gr.Blocks(title="Real Estate Investment RL") as demo:
        gr.Markdown("# 🏠 Real Estate Investment RL")
        gr.Markdown("Simple Gradio interface without audio dependencies.")

        with gr.Row():
            seed_input = gr.Number(label="Seed (optional)", value=None)
            reset_button = gr.Button("Reset")

        action_input = gr.Textbox(
            label="Action (5 values 0-4, comma-separated)",
            value="0,0,0,0,0",
        )
        step_button = gr.Button("Step")
        output_box = gr.Markdown()

        reset_button.click(reset_environment, inputs=[seed_input], outputs=[output_box])
        step_button.click(take_action, inputs=[action_input], outputs=[output_box])

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)