import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from multi_drone_mujoco.envs.hover_aviary import BaseAviary


import time

# Khởi tạo môi trường với gui=True hoặc render_mode="human"
env = BaseAviary(gui=True, render_mode="human") # hoặc class con kế thừa từ BaseAviary

obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Hoặc action từ model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # BẮT BUỘC CÓ DÒNG NÀY ĐỂ MỞ VÀ CẬP NHẬT CỬA SỔ MUJOCO
    env.render()
    
    # Delay nhẹ để không bị quá nhanh
    time.sleep(env.CTRL_TIMESTEP)

    if terminated or truncated:
        obs, info = env.reset()

env.close()