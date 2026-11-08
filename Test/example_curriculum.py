import os
import sys
import numpy as np

# Thêm thư mục gốc vào đường dẫn hệ thống để import multi_drone_mujoco
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.wrappers.curriculum import CurriculumWrapper, CurriculumConfig


# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA HÀM ĐIỀU CHỈNH ĐỘ KHÓ THEO TỪNG LEVEL
# ==============================================================================
def my_difficulty_fn(env, level):
    """Hàm này sẽ được gọi TỰ ĐỘNG ở mỗi đầu hiệp (hàm reset).
    
    Tùy vào `level` hiện tại, bạn có thể thay đổi bất kỳ thuộc tính nào của env:
    - Độ cao đích (Target Height)
    - Tốc độ gió (Wind Speed)
    - Số lượng vật cản, v.v.
    """
    # 1. Tăng dần độ cao mục tiêu mà drone cần bay tới
    target_height = 0.5 + level * 0.3
    env.TARGET_HEIGHT = target_height

    # 2. Level 0, 1: Không có gió (0 m/s)
    #    Level 2 trở lên: Bắt đầu có gió thổi tăng dần
    if level >= 2:
        wind_speed = (level - 1) * 0.8
    else:
        wind_speed = 0.0

    print(f"\n---> [difficulty_fn được gọi]: Đang ở LEVEL {level}")
    print(f"     + Độ cao mục tiêu: {target_height:.2f} m")
    print(f"     + Sức gió: {wind_speed:.2f} m/s")


# ==============================================================================
# BƯỚC 2: CẤU HÌNH BỘ QUY TẮC LÊN / XUỐNG LỚP (CURRICULUM CONFIG)
# ==============================================================================
# Để bạn dễ quan sát kết quả nhanh chóng, ta cài window_size nhỏ (chỉ 4 hiệp)
config = CurriculumConfig(
    metric="success_rate",      # Đánh giá dựa trên tỷ lệ thành công về đích
    threshold_advance=0.75,     # Nếu đạt >= 75% thành công -> LÊN CẤP (Level + 1)
    threshold_retreat=0.25,     # Nếu tụt xuống <= 25% thành công -> XUỐNG CẤP (Level - 1)
    window_size=4,              # Xét trung bình cứ mỗi 4 hiệp bay
    num_levels=5,               # Tổng cộng có 5 cấp độ (từ Level 0 đến 4)
    start_level=0,              # Bắt đầu từ Level 0 (dễ nhất)
    advance_count=1             # Đạt chuẩn 1 chu kỳ là cho lên cấp luôn
)


# ==============================================================================
# BƯỚC 3: BỌC MÔI TRƯỜNG VỚI CURRICULUM WRAPPER
# ==============================================================================
# Khởi tạo môi trường HoverAviary gốc (chế độ không cần GUI để chạy nhanh)
base_env = HoverAviary(gui=False)

# Bọc môi trường gốc vào CurriculumWrapper
env = CurriculumWrapper(base_env, difficulty_fn=my_difficulty_fn, config=config)


# ==============================================================================
# BƯỚC 4: CHẠY MÔ PHỎNG VÀ QUAN SÁT CƠ CHẾ TỰ ĐỘNG THĂNG / HẠ CẤP
# ==============================================================================
print("=" * 70)
print("BẮT ĐẦU MÔ PHỎNG MINH HỌA CURRICULUM LEARNING")
print("=" * 70)

total_episodes = 12

for episode in range(1, total_episodes + 1):
    # Mỗi lần gọi reset(), CurriculumWrapper sẽ tự kiểm tra và gọi difficulty_fn
    obs, info = env.reset()
    
    print(f"\n[HIỆP {episode} BẮT ĐẦU] Current Level: {info['curriculum_level']} (Tiến độ: {info['curriculum_progress']*100:.0f}%)")

    # Giả lập kết quả của Drone: 
    # - Hiệp 1 đến 8: Drone bay rất giỏi và thành công liên tục (để xem nó THĂNG CẤP)
    # - Hiệp 9 đến 12: Gặp bài khó hơn, drone bay hỏng liên tục (để xem nó HẠ CẤP)
    simulated_success = True if episode <= 8 else False

    # Chạy các bước bay trong hiệp
    for step in range(5):
        action = env.action_space.sample()  # Lấy hành động ngẫu nhiên
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Ở bước cuối, giả lập drone chạm đích (hoặc thất bại)
        if step == 4:
            # Gán cờ thành công trực tiếp vào wrapper để minh họa
            env._episode_success = simulated_success
            # Gọi hàm cập nhật hiệp bay của CurriculumWrapper
            env._record_episode(step_info)
            env._maybe_adjust_level()
            break

    # In kết quả sau khi kết thúc hiệp
    ket_qua = "THÀNH CÔNG (Về đích)" if simulated_success else "THẤT BẠI (Đâm va)"
    print(f"-> Kết thúc hiệp {episode}: {ket_qua}")
    
    # In thống kê cửa sổ đánh giá hiện tại
    stats = env.get_stats()
    print(f"   [Thống kê]: Tỷ lệ thành công gần đây: {stats['metric_avg']*100:.1f}% | Level hiện tại: {stats['level']}")

env.close()
print("\n" + "=" * 70)
print("MÔ PHỎNG HOÀN TẤT! Bạn thấy Level đã tự động tăng/giảm theo kết quả bay.")
print("=" * 70)
