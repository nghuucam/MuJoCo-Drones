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

# 3. Số lượng vật cản theo từng Cấp độ (Curriculum Obstacle Counts)
# Level 0: 3 cột trụ
# Level 1: 5 cột trụ
# Level 2: 7 cột trụ
# Level 3: 10 cột trụ
OBSTACLE_COUNTS = {
    0: 3,
    1: 5,
    2: 7,
    3: 10,
}

# Tọa độ cố định của 10 cột trụ tiềm năng trên bản đồ
OBSTACLE_POSITIONS = [
    [4.43, 7.36], [0.00, 7.00], [5.00, -1.00], [-4.46, -6.94], [2.37, 2.46],
    [0.00, -3.00], [-5.00, 7.00], [3.00, -7.00], [-5.00, 0.50], [-3.00, 4.00]
]

# 4. Thông số Cột trụ & Gai xương rồng (Cactus Obstacles & Spikes)
CYLINDER_RADIUS = 0.5            # Bán kính thân cột chính (m)
MIN_OBSTACLE_HEIGHT = 1.0        # Chiều cao tối thiểu cột trụ (m)
MAX_OBSTACLE_HEIGHT = 3.0        # Chiều cao tối đa cột trụ (m)

# Gai xương rồng (Spikes)
SPIKES_PER_PILLAR_MIN = 3        # Số gai tối thiểu ngẫu nhiên mỗi cột
SPIKES_PER_PILLAR_MAX = 6        # Số gai tối đa ngẫu nhiên mỗi cột
SPIKE_RADIUS_MIN = 0.05          # Bán kính đáy gai tối thiểu (m)
SPIKE_RADIUS_MAX = 0.20          # Bán kính đáy gai tối đa (m)
SPIKE_LENGTH_MIN = 0.20          # Độ dài gai nhô ra tối thiểu (m)
SPIKE_LENGTH_MAX = 0.80          # Độ dài gai nhô ra tối đa (m)

# 5. Cấu hình Bước & Giới hạn An toàn
MAX_STEPS = 80                   # Tối đa 80 bước cấp cao PPO mỗi episode
MAX_STEP_DISTANCE = 2.0          # Khoảng cách bước d in [0.1, 2.0] m
SUBSTEPS_PER_ACTION = 150        # Số bước PID con tối đa mỗi PPO step (~48Hz)

MIN_Z = 0.8                      # Độ cao tối thiểu an toàn (Cách đất >= 0.8m)
MAX_Z = 2.5                      # Độ cao tối đa
MAP_LIMIT_X = 6.5                # Giới hạn biên trục X
MAP_LIMIT_Y = 12.0               # Giới hạn biên trục Y

# 6. Hệ số va chạm mới (Collision Margin)
COLLISION_MARGIN = 0.05          # Giảm xuống 0.05m (5cm) giúp drone lách mượt quanh các gai nhọn

# 7. Kích thước ảnh camera FPV
IMG_WIDTH = 64
IMG_HEIGHT = 64
