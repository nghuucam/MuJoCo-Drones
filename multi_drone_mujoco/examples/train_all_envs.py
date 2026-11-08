import os
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import all environments
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
from multi_drone_mujoco.envs.race_aviary import RaceAviary
from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary
from multi_drone_mujoco.envs.formation_aviary import FormationAviary
# from multi_drone_mujoco.envs.multi_agent_aviary import MultiAgentAviary

def train_env(env_name, env_class, env_kwargs=None, algo=PPO, total_timesteps=10000000, n_envs=4):
    """
    Generic function to spin up vectorized environments and train an RL model.
    """
    if env_kwargs is None:
        env_kwargs = {"gui": False, "record": False}
        
    print(f"\n{'='*60}")
    print(f"Starting Training Suite for: {env_name}")
    print(f"{'='*60}")
    
    # Create vectorized environments for faster data collection
    def make_env():
        return env_class(**env_kwargs)
        
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    eval_env = env_class(**env_kwargs)
    
    log_dir = f"./logs/{env_name}_{int(time.time())}/"
    os.makedirs(log_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True,
        render=False
    )
    
    # Initialize the model
    # Note: For massive training runs on a cluster, you might want to uncomment tensorboard logging (requires pip install tensorboard)
    model = algo("MlpPolicy", vec_env, verbose=1)
    
    print(f"Training {env_name} for {total_timesteps} steps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        print(f"Training interrupted for {env_name}. Saving current model...")
    
    model.save(f"{log_dir}/final_model")
    vec_env.close()
    eval_env.close()
    print(f"Finished {env_name}.")

def main():
    # Setup the environments to train.
    # Note: MultiAgentAviary is excluded here because it uses PettingZoo ParallelEnv
    # and requires a MARL-specific wrapper (e.g. SuperSuit) to train with SB3.
    # The other multi-drone envs use Centralized Control (a single large action space).
    envs_to_train = [
        ("HoverAviary", HoverAviary, {}),
        ("VelocityAviary", VelocityAviary, {}),
        ("FlyThroughAviary", FlyThroughAviary, {}),
        ("RaceAviary", RaceAviary, {"num_drones": 1}),
        ("MultiHoverAviary", MultiHoverAviary, {"num_drones": 2}),
        ("FormationAviary", FormationAviary, {"num_drones": 3}),
    ]
    
    # For a full scale run, you should set steps to 10M-50M.
    # For testing/smoke tests, set this to 5,000 steps.
    STEPS_PER_ENV = 5000 
    
    for name, cls, kwargs in envs_to_train:
        # We use PPO as it is highly stable for continuous control drones
        train_env(name, cls, kwargs, algo=PPO, total_timesteps=STEPS_PER_ENV, n_envs=2)
        
    print("All configured environments have finished their training loops!")

if __name__ == "__main__":
    main()
