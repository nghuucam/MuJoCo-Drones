"""Example: Play/visualize a trained policy.

Usage:
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip
"""

import argparse
import numpy as np


def play(model_path: str, env_type: str = "hover", episodes: int = 3):
    """Load and visualize a trained policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    if env_type == "multi":
        env = MultiHoverAviary(num_drones=2, ctrl_freq=48, sim_freq=240, render_mode="rgb_array")
    else:
        env = HoverAviary(ctrl_freq=48, sim_freq=240, render_mode="rgb_array")

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes)
