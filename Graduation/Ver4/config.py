# ==============================================================================
# CẤU HÌNH CHO MÔI TRƯỜNG DRONE PPO CURRICULUM (GRADUATION / VER4)
# ==============================================================================

# 1. Điểm xuất phát (Xung quanh Y = 11.0m, Z = 1.0m)
start_space = [
    [0.0, 11.0], [1.0, 11.0], [2.0, 11.0], [-1.0, 11.0], [-2.0, 11.0],
    [3.0, 11.0], [4.0, 11.0], [-3.0, 11.0], [-4.0, 11.0]
]

# 2. Vị trí Đích đến theo từng Cấp độ (Curriculum Goal Y Ranges)
# - Level 0: Giữ nguyên như cũ: Y ~ [7.5, 8.5] (Quãng đường ~ 3m)
# - Level 1: Tăng gấp đôi độ dài: Y ~ [-0.5, 0.5] (Quãng đường ~ 11m, gấp đôi 5.5m)
# - Level 2: Tăng gấp đôi độ dài: Y ~ [-5.5, -4.5] (Quãng đường ~ 16m, gấp đôi 8.0m)
# - Level 3: Tăng gấp đôi độ dài: Y ~ [-10.5, -9.5] (Quãng đường ~ 21m, gấp đôi 10.5m)
GOAL_Y_RANGES = {
    0: (7.5, 8.5),
    1: (-0.5, 0.5),
    2: (-5.5, -4.5),
    3: (-10.5, -9.5),
}

# Bán kính vùng đích: Giảm từ 2.0m xuống 1.0m (đòi hỏi độ chính xác cao khi tiếp cận đích)
GOAL_THRESHOLD = 1.0             # Bán kính vùng đích (m): Drone cách tâm đích < 1.0m là tính tới đích thành công

# 3. Số lượng vật cản theo từng Cấp độ (Tăng gấp đôi từ Level 1 -> 3)
# - Level 0: 0 cột trụ (Giữ nguyên như cũ, tập cất cánh & bay thẳng)
# - Level 1: 6 cột trụ (Gấp đôi Ver3: 3 -> 6 cột)
# - Level 2: 10 cột trụ (Gấp đôi Ver3: 5 -> 10 cột)
# - Level 3: 14 cột trụ (Gấp đôi Ver3: 7 -> 14 cột)
OBSTACLE_COUNTS = {
    0: 0,
    1: 6,
    2: 10,
    3: 14,
}

# Tọa độ 14 cột trụ trên bản đồ mở rộng Y (từ Y=11m xuống Y=-10m):
OBSTACLE_POSITIONS = [
    # --- Level 1 (6 cột đầu tiên: trải dài từ Y = 7.5m xuống Y = 2.0m) ---
    [-4.50, 7.50],   # Cột 1: Hàng 1 - Trái
    [ 0.00, 7.00],   # Cột 2: Hàng 1 - Giữa
    [ 4.50, 7.50],   # Cột 3: Hàng 1 - Phải
    [-2.50, 4.00],   # Cột 4: Hàng 2 - So le trái
    [ 2.50, 4.00],   # Cột 5: Hàng 2 - So le phải
    [ 0.00, 2.00],   # Cột 6: Hàng 3 - Chặn trung tâm trước mốc đích Level 1

    # --- Level 2 (Bổ sung thêm 4 cột = 10 cột: trải dài từ Y = 0.5m xuống Y = -2.5m) ---
    [-4.00, 0.50],   # Cột 7: Sườn trái mốc Y = 0.5m
    [ 4.00, 0.50],   # Cột 8: Sườn phải mốc Y = 0.5m
    [-2.20, -2.50],  # Cột 9: So le trái trước mốc đích Level 2
    [ 2.20, -2.50],  # Cột 10: So le phải trước mốc đích Level 2

    # --- Level 3 (Bổ sung thêm 4 cột = 14 cột: trải dài từ Y = -5.0m xuống Y = -8.0m) ---
    [-4.20, -5.00],  # Cột 11: Trái hàng 5
    [ 0.00, -5.50],  # Cột 12: Giữa hàng 5
    [ 4.20, -5.00],  # Cột 13: Phải hàng 5
    [-2.00, -8.00],  # Cột 14: Chặn trung tâm trước mốc đích Level 3 (-10m)
]

# 4. Thông số Cột trụ & Gai xương rồng (Cactus Obstacles & Spikes)
CYLINDER_RADIUS = 0.5            # Bán kính thân cột chính (m)
FIXED_OBSTACLE_HEIGHT = 3.0      # Chiều cao vật cản cố định cho tất cả các level (m)
MIN_OBSTACLE_HEIGHT = 3.0        # (Tương thích ngược)
MAX_OBSTACLE_HEIGHT = 3.0        # (Tương thích ngược)

# Số lượng gai trên mỗi cột trụ: TĂNG GẤP ĐÔI (từ 3-6 lên 6-12 gai mỗi cột)
SPIKES_PER_PILLAR_MIN = 6        # Số gai tối thiểu ngẫu nhiên mỗi cột (Gấp đôi: 3 -> 6)
SPIKES_PER_PILLAR_MAX = 12       # Số gai tối đa ngẫu nhiên mỗi cột (Gấp đôi: 6 -> 12)
SPIKE_RADIUS_MIN = 0.05          # Bán kính đáy gai tối thiểu (m)
SPIKE_RADIUS_MAX = 0.20          # Bán kính đáy gai tối đa (m)
SPIKE_LENGTH_MIN = 0.20          # Độ dài gai nhô ra tối thiểu (m)
SPIKE_LENGTH_MAX = 0.80          # Độ dài gai nhô ra tối đa (m)

# 5. Cấu hình Bước & Giới hạn Bản đồ Động (Dynamic Map Boundaries)
# Quãng đường tối đa tăng từ 10.5m lên 21m -> Tăng MAX_STEPS từ 80 lên 140
MAX_STEPS = 140                  # Tối đa 140 bước cấp cao PPO mỗi episode cho bản đồ dài 21m
MAX_STEP_DISTANCE = 2.0          # Khoảng cách bước d in [0.1, 2.0] m
SUBSTEPS_PER_ACTION = 150        # Số bước PID con tối đa mỗi PPO step (~48Hz)

OVERMAP_PAST_GOAL_DISTANCE = 0.0     # Đúng bằng 0: Không cho phép vượt quá vị trí đích (Y không được nhỏ hơn Y_goal)
OVERMAP_BEHIND_START_DISTANCE = 1.0  # Tối đa 1 đơn vị sau điểm xuất phát Y
OVERMAP_X_MARGIN = 1.5               # Lề đệm vượt ra ngoài cột ngoài cùng trục X

MAP_LIMIT_X = 6.5                # Giới hạn biên trục X tối đa toàn map
MAP_LIMIT_Y = 22.0               # Mở rộng giới hạn biên trục Y từ 12.0m lên 22.0m (bao phủ Y = -10.0m)
MIN_Z = 0.8                      # Độ cao tối thiểu an toàn (Cách đất >= 0.8m)
MAX_Z = 2.5                      # Độ cao tối đa

# 6. Hệ số va chạm (Collision Margin)
COLLISION_MARGIN = 0.05          # Giữ nguyên 0.05m (5cm) giúp drone lách mượt quanh các gai nhọn

# 7. Kích thước ảnh & Cấu hình Camera FPV
IMG_WIDTH = 64
IMG_HEIGHT = 64
CAMERA_FOVY = 75.0               # Góc mở camera FPV (độ). Ảnh 64x64 nên góc ngang = góc dọc = 75 độ
CAMERA_HALF_FOV = 37.5           # Nửa góc mở = 37.5 độ (Biên tối đa camera nhìn thấy mỗi bên)

# Khống chế góc bay (alpha, beta) nằm gọn bên trong vùng nhìn thấy của camera
MAX_ALPHA_DEG = 35.0             # Góc lái ngang alpha in [-35 deg, +35 deg] (nằm trong tầm nhìn 37.5 deg)
MAX_BETA_DEG = 30.0              # Góc nâng/chúc beta in [-30 deg, +30 deg] (nằm trong tầm nhìn 37.5 deg)

# 8. Cấu hình Camera 3D GUI khi hiển thị trực tiếp (GUI Viewer Camera)
GUI_CAMERA_TRACKING = True     # True: Tự động khóa camera đi theo Drone (Tracking mode)
                               # False: Camera tự do đứng yên một chỗ (Free mode)
GUI_CAMERA_DISTANCE = 3.5      # Khoảng cách từ camera tới Drone (m)
GUI_CAMERA_ELEVATION = -20.0   # Góc ngẩng/chúc nhìn từ trên xuống (độ, ví dụ: -20 đến -30)
GUI_CAMERA_AZIMUTH = -90.0     # Góc xoay ngang (-90 là nhìn từ sau lưng Drone thẳng về hướng đích)
GUI_SHOW_RIGHT_UI = False      # False: Ẩn thanh công cụ bên phải để màn hình rộng rãi
GUI_SHOW_LEFT_UI = False       # False: Ẩn thanh thông số bên trái

# 9. Cấu hình Cửa sổ hiển thị kép (Dual Window: MuJoCo 3D + Drone POV FPV)
GUI_SHOW_POV_WINDOW = True     # True: Mở thêm cửa sổ thứ 2 hiển thị trực tiếp camera POV (FPV) từ mũi Drone
GUI_POV_WINDOW_SIZE = 256      # Kích thước cửa sổ POV (pixels, ví dụ: 256x256 hoặc 320x320)
