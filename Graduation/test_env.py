import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_env import DronePPOEnv
import config
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

def test_environment_basic():
    print("=" * 80)
    print("🧪 KIỂM TRA 1: TƯƠNG THÍCH VỚI GYMNASIUM & STABLE-BASELINES3 CHECK_ENV")
    print("=" * 80)
    
    env = DronePPOEnv(gui=False)
    
    # Kiểm tra chuẩn Gymnasium / SB3
    try:
        check_env(env, warn=True)
        print("✅ Môi trường vượt qua hoàn toàn kiểm tra của check_env (SB3)!")
    except Exception as e:
        print(f"⚠️ Cảnh báo từ check_env: {e}")

    print("\n" + "=" * 80)
    print("🧪 KIỂM TRA 2: RESET & RANDOM CHIỀU CAO CỘT VẬT CẢN [0.1m - 3.0m]")
    print("=" * 80)
    
    obs, info = env.reset()
    print(f"📍 Điểm xuất phát: {env.start[0][:2]} | Điểm đích: {env.goal[0][:2]}")
    print(f"📸 Kích thước ảnh FPV nhận được: {obs.shape}, kiểu dữ liệu: {obs.dtype}")
    print(f"🚁 Vị trí Drone sau khi cất cánh hover: {env.pos[0]}")
    
    print("\n--- Chiều cao 10 cột được sinh ngẫu nhiên ---")
    for i, obs_info in enumerate(env.obstacle_data):
        print(f"  Cột {i}: Tọa độ=({obs_info[0]:.2f}, {obs_info[1]:.2f}), Chiều cao H={obs_info[2]:.2f}m")

    print("\n" + "=" * 80)
    print("🧪 KIỂM TRA 3: HÀNH ĐỘNG PPO (alpha, beta, d) & CHẶNG CỨNG CHỐNG ĐÂM ĐẤT")
    print("=" * 80)

    # Thử nghiệm 1: Bay thẳng giữ nguyên độ cao (alpha=90°, beta=90°, d=1.5m)
    action_forward = np.array([90.0, 90.0, 1.5], dtype=np.float32)
    print(f"\n👉 Gửi Action 1 (Đi thẳng): alpha=90°, beta=90°, d=1.5m")
    obs, r, term, trunc, info = env.step(action_forward)
    print(f"   -> Điểm đến mục tiêu: {info['target_pos']}")
    print(f"   -> Vị trí thực tế drone: {env.pos[0]}")
    print(f"   -> Reward: {r:.3f}, Terminated: {term}, Truncated: {trunc}")

    # Thử nghiệm 2: Cố tình chúc mũi cắm xuống đất (beta=10° < 90°, d=2.0m)
    action_dive = np.array([90.0, 10.0, 2.0], dtype=np.float32)
    print(f"\n👉 Gửi Action 2 (Cố tình chúc xuống đất): alpha=90°, beta=10°, d=2.0m")
    obs, r, term, trunc, info = env.step(action_dive)
    print(f"   -> Điểm đến mục tiêu: {info['target_pos']} (Phải được chặn cứng >= {config.MIN_Z}m)")
    print(f"   -> Vị trí thực tế drone: {env.pos[0]}")
    assert info['target_pos'][2] >= config.MIN_Z, "LỖI: Chưa chặn cứng được độ cao tối thiểu!"
    print(f"   ✅ Chặn cứng thành công! Độ cao mục tiêu z={info['target_pos'][2]:.2f}m >= {config.MIN_Z}m")

    # Thử nghiệm 3: Rẽ phải FPV (alpha=45°, beta=90°, d=1.0m)
    action_right = np.array([45.0, 90.0, 1.0], dtype=np.float32)
    print(f"\n👉 Gửi Action 3 (Lượn phải): alpha=45°, beta=90°, d=1.0m")
    obs, r, term, trunc, info = env.step(action_right)
    print(f"   -> Điểm đến mục tiêu: {info['target_pos']}")
    print(f"   -> Vị trí thực tế drone: {env.pos[0]}")

    # Lưu ảnh quan sát FPV ra file để kiểm tra trực quan
    out_img_path = os.path.join(current_dir, "fpv_camera_sample.png")
    plt.imsave(out_img_path, obs)
    print(f"\n🖼️ Đã lưu mẫu ảnh FPV camera vào: {out_img_path}")

    env.close()
    print("\n🎉 TẤT CẢ KIỂM TRA ĐÃ HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    test_environment_basic()
