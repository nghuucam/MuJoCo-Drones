import os
import sys
import numpy as np
from stable_baselines3.common.env_checker import check_env

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv


def test_environment_api():
    print("=" * 80)
    print("🔍 KIỂM TRA TÍNH TƯƠNG THÍCH MÔI TRƯỜNG VER3 (MULTI-INPUT) VỚI SB3 (CHECK_ENV)")
    print("=" * 80)

    env = DronePPOCurriculumEnv(gui=False)

    print("1. Chạy check_env của Stable-Baselines3...")
    check_env(env, warn=True)
    print("✅ Môi trường vượt qua bài kiểm tra SB3 check_env 100%!")

    print("\n2. Kiểm tra chu trình Reset & Step ở tất cả 4 Level...")
    for lvl in range(4):
        env.set_level(lvl)
        obs, info = env.reset()
        assert isinstance(obs, dict), f"Lỗi: obs phải là Dict, nhận được {type(obs)}"
        assert "rgb" in obs and "state" in obs, "Thiếu key 'rgb' hoặc 'state' trong obs!"
        assert obs["rgb"].shape == (64, 64, 3), f"Lỗi shape ảnh FPV: {obs['rgb'].shape}"
        assert obs["rgb"].dtype == np.uint8, f"Lỗi dtype ảnh FPV: {obs['rgb'].dtype}"
        assert obs["state"].shape == (18,), f"Lỗi shape vector state: {obs['state'].shape}"
        assert obs["state"].dtype == np.float32, f"Lỗi dtype vector state: {obs['state'].dtype}"
        assert "is_success" in info, "Thiếu cờ is_success trong info!"

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        alpha_h = next_obs['state'][16]
        beta_h = next_obs['state'][17]
        print(f"   - Level {lvl}: Reset OK | Step OK | Reward = {reward:.2f} | Pos = {next_obs['state'][:3].round(2)} | Goal = {next_obs['state'][9:12].round(2)} | APF Hint [α={alpha_h:+.2f}, β={beta_h:+.2f}]")

    env.close()
    print("\n🎉 HOÀN THÀNH TOÀN BỘ KIỂM THỬ MÔI TRƯỜNG VER3!")


if __name__ == "__main__":
    test_environment_api()
