# ==============================================================================
# CẤU HÌNH CHO MÔI TRƯỜNG DRONE PPO CURRICULUM (GRADUATION / VER2)
# ==============================================================================

# 1. Điểm xuất phát (Giữ nguyên xung quanh Y = 11.0m)
start_space = [
    [0.0, 11.0], [1.0, 11.0], [2.0, 11.0], [-1.0, 11.0], [-2.0, 11.0],
    [3.0, 11.0], [4.0, 11.0], [-3.0, 11.0], [-4.0, 11.0]
]

# 2. Vị trí Đích đến theo từng Cấp độ (Curriculum Goal Y Ranges - Không vượt quá Y = 0)
# Level 0 (Cấp 1): Y ~ [7.5, 8.5] (Quãng đường ~ 3m)
# Level 1 (Cấp 2): Y ~ [5.0, 6.0] (Quãng đường ~ 5.5m)
# Level 2 (Cấp 3): Y ~ [2.5, 3.5] (Quãng đường ~ 8m)
# Level 3 (Cấp 4): Y ~ [0.0, 1.0] (Quãng đường ~ 10.5m - Chạm mốc Y=0)
GOAL_Y_RANGES = {
    0: (7.5, 8.5),
    1: (5.0, 6.0),
    2: (2.5, 3.5),
    3: (0.0, 1.0),
}
GOAL_THRESHOLD = 2.0             # Bán kính vùng đích (m): Drone cách tâm đích < 2.0m là tính tới đích thành công

# 3. Số lượng vật cản theo từng Cấp độ (Curriculum Obstacle Counts)
# Level 0: 0 cột trụ (Hoàn toàn trống trải, học cất cánh & bay thẳng tới đích)
# Level 1: 3 cột trụ (Hàng rào đầu tiên chặn trước đích)
# Level 2: 5 cột trụ (Thêm 2 cột so le, học lượn chữ S)
# Level 3: 7 cột trụ (Thêm 2 cột kẹp sườn bản đồ)
OBSTACLE_COUNTS = {
    0: 0,
    1: 3,
    2: 5,
    3: 7,
}

# Tọa độ cố định của các cột trụ trên bản đồ (sắp xếp theo tiến trình Curriculum):
# Level 1 (3 cột): [-5.00, 7.00], [0.00, 7.00], [4.43, 7.36] (Trái, Giữa, Phải)
# Level 2 (5 cột): Thêm 2 cột ở dải Y ~ [2.5, 4.0]: [-3.00, 4.00], [2.37, 2.46]
# Level 3 (7 cột): Thêm 2 cột ở dải Y ~ [-1.0, 0.5]: [-5.00, 0.50], [5.00, -1.00]
OBSTACLE_POSITIONS = [
    [-5.00, 7.00], [0.00, 7.00], [4.43, 7.36],   # Level 1 (3 cột: Trái, Giữa, Phải)
    [-3.00, 4.00], [2.37, 2.46],                 # Level 2 (+2 cột = 5 cột)
    [-5.00, 0.50], [5.00, -1.00],                 # Level 3 (+2 cột = 7 cột)
    [0.00, -3.00], [-4.46, -6.94], [3.00, -7.00] # Dự phòng
]

# 4. Thông số Cột trụ & Gai xương rồng (Cactus Obstacles & Spikes)
CYLINDER_RADIUS = 0.5            # Bán kính thân cột chính (m)
FIXED_OBSTACLE_HEIGHT = 3.0      # Chiều cao vật cản cố định cho tất cả các level (m) - Không biến thiên ngẫu nhiên
MIN_OBSTACLE_HEIGHT = 3.0        # (Tương thích ngược)
MAX_OBSTACLE_HEIGHT = 3.0        # (Tương thích ngược)

# Gai xương rồng (Spikes)
SPIKES_PER_PILLAR_MIN = 3        # Số gai tối thiểu ngẫu nhiên mỗi cột
SPIKES_PER_PILLAR_MAX = 6        # Số gai tối đa ngẫu nhiên mỗi cột
SPIKE_RADIUS_MIN = 0.05          # Bán kính đáy gai tối thiểu (m)
SPIKE_RADIUS_MAX = 0.20          # Bán kính đáy gai tối đa (m)
SPIKE_LENGTH_MIN = 0.20          # Độ dài gai nhô ra tối thiểu (m)
SPIKE_LENGTH_MAX = 0.80          # Độ dài gai nhô ra tối đa (m)

# 5. Cấu hình Bước & Giới hạn Bản đồ Động (Dynamic Map Boundaries)
MAX_STEPS = 80                   # Tối đa 80 bước cấp cao PPO mỗi episode
MAX_STEP_DISTANCE = 2.0          # Khoảng cách bước d in [0.1, 2.0] m
SUBSTEPS_PER_ACTION = 150        # Số bước PID con tối đa mỗi PPO step (~48Hz)

OVERMAP_PAST_GOAL_DISTANCE = 0.0     # Đúng bằng 0: Không cho phép vượt quá vị trí đích (Y không được nhỏ hơn Y_goal)
OVERMAP_BEHIND_START_DISTANCE = 1.0  # Tối đa 1 đơn vị sau điểm xuất phát Y
OVERMAP_X_MARGIN = 1.5               # Lề đệm vượt ra ngoài cột ngoài cùng trục X

MAP_LIMIT_X = 6.5                # Giới hạn biên trục X tối đa toàn map
MAP_LIMIT_Y = 12.0               # Giới hạn biên trục Y tối đa toàn map
MIN_Z = 0.8                      # Độ cao tối thiểu an toàn (Cách đất >= 0.8m)
MAX_Z = 2.5                      # Độ cao tối đa

# 6. Hệ số va chạm mới (Collision Margin)
COLLISION_MARGIN = 0.05          # Giảm xuống 0.05m (5cm) giúp drone lách mượt quanh các gai nhọn

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
GUI_SHOW_RIGHT_UI = False      # False: Ẩn thanh công cụ bên phải (Joint, Control, Equality) để màn hình rộng rãi
GUI_SHOW_LEFT_UI = False       # False: Ẩn thanh thông số bên trái

# 9. Cấu hình Cửa sổ hiển thị kép (Dual Window: MuJoCo 3D + Drone POV FPV)
GUI_SHOW_POV_WINDOW = True     # True: Mở thêm cửa sổ thứ 2 hiển thị trực tiếp camera POV (FPV) từ mũi Drone
GUI_POV_WINDOW_SIZE = 256      # Kích thước cửa sổ POV (pixels, ví dụ: 256x256 hoặc 320x320)
