import os
import sys
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv


def main():
    print("=" * 80)
    print("🌵 VER2 TEST GUI: KIỂM TRA MÔI TRƯỜNG CỘT TRỤ GAI XƯƠNG RỒNG & CURRICULUM")
    print("=" * 80)
    print("🎮 Đang mở giao diện 3D MuJoCo...")

    env = DronePPOCurriculumEnv(gui=True)

    # Chạy thử lần lượt từ Level 0 đến Level 3
    for level in range(4):
        print(f"\n" + "=" * 60)
        print(f"🎯 ĐÃ THIẾT LẬP CURRICULUM LEVEL {level}")
        print("=" * 60)
        env.set_level(level)
        obs, info = env.reset()

        print(f"📍 Số cột cản kích hoạt: {len(env.obstacle_data)} cột trụ gai xương rồng")
        print(f"🏁 Vị trí Đích màu đỏ: X={env.goal[0,0]:.2f}, Y={env.goal[0,1]:.2f}, Z={env.goal[0,2]:.2f}")
        print("💡 MẸO: Bạn có thể nhấn [SPACE] trên cửa sổ 3D bất kỳ lúc nào để Pause/Resume!")
        input("\n⏸️ Nhấn phím [ENTER] trong terminal này để bắt đầu cho Drone bay...")

        for step in range(60):
            # Chọn hành động bay tiến nhẹ về phía trước
            action = np.array([0.0, 0.0, 0.3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            # Giảm tốc độ mô phỏng (0.15s mỗi bước) để mắt người dễ dàng quan sát
            time.sleep(0.15)

            if terminated or truncated:
                print(f"🏁 Kết thúc Episode Level {level} ở bước {step + 1}! (Win: {info['win']}, Collision: {info['collision']})")
                break

        if level < 3:
            input("\n⏸️ Nhấn phím [ENTER] để tiếp tục sang Level tiếp theo...")

    print("\n🎉 Hoàn tất xem thử giao diện 3D!")
    input("⏸️ Nhấn [ENTER] lần cuối để đóng cửa sổ mô phỏng...")
    env.close()


if __name__ == "__main__":
    main()
