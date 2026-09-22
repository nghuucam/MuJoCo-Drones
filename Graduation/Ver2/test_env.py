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
    print("🔍 KIỂM TRẢ TÍNH TƯƠNG THÍCH MÔI TRƯỜNG VER2 VỚI STABLE-BASELINES3 (CHECK_ENV)")
    print("=" * 80)

    env = DronePPOCurriculumEnv(gui=False)

    print("1. Chạy check_env của Stable-Baselines3...")
    check_env(env, warn=True)
    print("✅ Môi trường vượt qua bài kiểm tra SB3 check_env 100%!")

    print("\n2. Kiểm tra chu trình Reset & Step ở tất cả 4 Level...")
    for lvl in range(4):
        env.set_level(lvl)
        obs, info = env.reset()
        assert obs.shape == (64, 64, 3), f"Lỗi shape quan sát ảnh FPV: {obs.shape}"
        assert obs.dtype == np.uint8, f"Lỗi dtype quan sát: {obs.dtype}"
        assert "is_success" in info, "Thiếu cờ is_success trong info!"

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        print(f"   - Level {lvl}: Reset OK | Step OK | Reward = {reward:.2f} | Obs shape = {next_obs.shape}")

    env.close()
    print("\n🎉 HOÀN THÀNH TOÀN BỘ KIỂM THỬ MÔI TRƯỜNG VER2!")


if __name__ == "__main__":
    test_environment_api()
