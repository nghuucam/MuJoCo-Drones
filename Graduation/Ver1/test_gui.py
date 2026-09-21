import os
import sys
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_env import DronePPOEnv
from stable_baselines3 import PPO
import config

def main():
    print("=" * 80)
    print("🎮 CHẠY THỬ NGHIỆM MÔ PHỎNG 3D TRỰC QUAN (GUI)")
    print("=" * 80)

    # Bật giao diện GUI MuJoCo để người dùng nhìn thấy drone bay trực quan
    env = DronePPOEnv(gui=True)
    obs, info = env.reset()

    # Kiểm tra xem có model đã huấn luyện trong thư mục models hay không
    model_path = os.path.join(current_dir, "models", "drone_ppo_final.zip")
    interrupted_path = os.path.join(current_dir, "models", "drone_ppo_interrupted.zip")
    
    model = None
    if os.path.exists(model_path):
        print(f"📥 Đang nạp model: {model_path}")
        model = PPO.load(model_path, env=env)
    elif os.path.exists(interrupted_path):
        print(f"📥 Đang nạp model checkpoint: {interrupted_path}")
        model = PPO.load(interrupted_path, env=env)
    else:
        print("ℹ️ Chưa có model huấn luyện. Đang chạy thử nghiệm với các hành động mẫu...")

    NUM_EPISODES = 5
    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n--- EPISODE {ep}/{NUM_EPISODES} ---")
        obs, info = env.reset()
        print(f"📍 Xuất phát: {env.start[0][:2]} | Đích: {env.goal[0][:2]}")

        for step in range(config.MAX_STEPS):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Nếu chưa có model, bay thẳng tới trước và né nhẹ
                # action in [-1, 1]: 0.0 là bay thẳng, >0 là rẽ trái, <0 là rẽ phải
                turn = np.random.uniform(-0.3, 0.3)
                pitch = np.random.uniform(-0.1, 0.1)
                dist = np.random.uniform(0.2, 0.8)
                action = np.array([turn, pitch, dist], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            cur_p = env.pos[0]
            print(f"  Bước {step+1}: Drone pos=[{cur_p[0]:.2f}, {cur_p[1]:.2f}, {cur_p[2]:.2f}] | Target={info['target_pos']}")

            if terminated or truncated:
                if info["is_success"]:
                    print("  🏆 THÀNH CÔNG TỚI ĐÍCH!")
                elif info["collision"]:
                    print("  💥 VA CHẠM CỘT HOẶC RƠI ĐẤT!")
                else:
                    print("  ⚠️ HẾT BƯỚC HOẶC RA NGOÀI BẢN ĐỒ!")
                break

    env.close()
    print("\n✅ Hoàn thành phiên thử nghiệm GUI!")

if __name__ == "__main__":
    main()
