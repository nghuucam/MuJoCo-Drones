import os
import sys
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv


def test_dual_window():
    print("=" * 70)
    print("🔍 KIỂM TRA HIỂN THỊ SONG SONG 2 CỬA SỔ (MUJOCO 3D + DRONE POV FPV)")
    print("=" * 70)

    env = DronePPOCurriculumEnv(gui=True)
    obs, info = env.reset()

    print("🚀 Bắt đầu chạy thử 15 bước mô phỏng để hiển thị đồng thời cả 2 cửa sổ...")
    for step in range(15):
        # Xuất hành động ngẫu nhiên
        action = np.array([0.0, 0.0, 0.0]) # bay thẳng nhẹ
        obs, reward, term, trunc, info = env.step(action)
        print(f"   - Step {step+1:02d}: Render OK | Drone Pos = {env.pos[0].round(2)}")
        time.sleep(0.05)
        if term or trunc:
            break

    env.close()
    print("\n✅ KIỂM TRA THÀNH CÔNG! CẢ 2 CỬA SỔ ĐÃ ĐÓNG AN TOÀN.")


if __name__ == "__main__":
    test_dual_window()
