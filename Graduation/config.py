# ==============================================================================
# CẤU HÌNH CHO MÔI TRƯỜNG DRONE PPO (GRADUATION)
# ==============================================================================

# Không gian điểm xuất phát và đích đến (Kế thừa từ môi trường D3QN cũ)
goal_space = [[0, -10], [1, -10], [2, -10], [-1, -10], [-2, -10], 
              [0, -11], [1, -11], [2, -11], [-1, -11], [-2, -11]]

start_space = [[0, 11], [1, 11], [2, 11], [-1, 11], [-2, 11], 
               [3, 11], [4, 11], [-3, 11], [-4, 11]]

obstacle_position = [[4.43, 7.36], [0.00, 7.00], [5.00, -1.00], [-4.46, -6.94], [2.37, 2.46], 
                     [0.00, -3.00], [-5.00, 7.00], [3.00, -7.00], [-5.00, 0.50], [-3.00, 4.00]]

# Cấu hình bước hành động (Action / Step Limits)
MAX_STEPS = 70                # Tối đa 70 bước quyết định cấp cao (PPO) mỗi episode
MAX_STEP_DISTANCE = 2.0       # Khoảng cách tối đa mỗi bước di chuyển d in [0, 2.0] mét
SUBSTEPS_PER_ACTION = 150     # Số bước điều khiển PID tầng thấp tối đa mỗi bước cấp cao (ở 48Hz ~ 3s)

# Giới hạn không gian an toàn (Hard Clamps)
MIN_Z = 0.4                   # Độ cao tối thiểu (Chống đâm xuyên đất)
MAX_Z = 2.5                   # Độ cao tối đa
MAP_LIMIT_X = 6.5             # Giới hạn biên map theo trục X
MAP_LIMIT_Y = 12.0            # Giới hạn biên map theo trục Y

# Kích thước vật cản ngẫu nhiên
CYLINDER_RADIUS = 1.0         # Bán kính cột trụ
MIN_OBSTACLE_HEIGHT = 0.1     # Chiều cao cột tối thiểu (m)
MAX_OBSTACLE_HEIGHT = 3.0     # Chiều cao cột tối đa (m)
COLLISION_DISTANCE = 0.3      # Khoảng cách báo va chạm tính từ mặt ngoài cột (bán kính + 0.3m)

# Kích thước ảnh camera FPV
IMG_WIDTH = 64
IMG_HEIGHT = 64
