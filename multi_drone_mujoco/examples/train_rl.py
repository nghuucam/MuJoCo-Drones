import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from multi_drone_mujoco.envs.hover_aviary import HoverAviary

def main():
    print("="*60)
    print("Training PPO on HoverAviary")
    print("="*60)

    # Create the training environment
    env = HoverAviary(record=False, gui=False)
    
    # Create the evaluation environment
    eval_env = HoverAviary(record=False, gui=False)
    
    # Setup evaluation callback to log reward progression
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Initialize the PPO model
    # We use a simple MLP policy. 
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train for 100,000 steps. This is relatively short but should show the reward trend.
    model.learn(total_timesteps=100000, callback=eval_callback)
    
    print("Training finished. Evaluating final policy...")
    
    # Evaluate a few episodes
    obs, info = eval_env.reset()
    total_reward = 0
    episodes = 0
    while episodes < 5:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode {episodes+1} finished with reward: {total_reward:.2f}")
            obs, info = eval_env.reset()
            total_reward = 0
            episodes += 1

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
