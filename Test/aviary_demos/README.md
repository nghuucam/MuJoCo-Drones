# Hướng dẫn Kiểm tra và Đánh giá 5 Môi trường Aviary trong MuJoCo-Drones

Thư mục này chứa toàn bộ các kịch bản kiểm tra (test scripts), trực quan hóa (visualization) và bộ điều khiển thử nghiệm cho 5 môi trường aviary nâng cao được xây dựng trên nền tảng vật lý MuJoCo:

1. **VelocityAviary** (Bám đuổi vận tốc 3D & vận tốc góc Yaw)
2. **FlyThroughAviary** (Bay tuần tự qua các điểm mốc Waypoints)
3. **FormationAviary** (Bay phối hợp đội hình hình học nhiều drone)
4. **RaceAviary** (Đua qua chuỗi các cổng Circuit Gates)
5. **MultiAgentAviary** (Chuẩn giao diện PettingZoo ParallelEnv Đa tác tử)

---

## 1. Cấu trúc thư mục

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos\
├── controller_utils.py             # Bộ chuyển đổi RPM sang Action [-1, 1], bộ phân tích tham số CLI, xuất GIF
├── 01_test_velocity_aviary.py      # Script test môi trường VelocityAviary
├── 02_test_fly_through_aviary.py   # Script test môi trường FlyThroughAviary
├── 03_test_formation_aviary.py     # Script test môi trường FormationAviary
├── 04_test_race_aviary.py          # Script test môi trường RaceAviary
├── 05_test_multi_agent_aviary.py   # Script test môi trường MultiAgentAviary
├── run_all.py                      # Script tổng hợp chạy tất cả hoặc từng môi trường
├── README.md                       # Tài liệu hướng dẫn chi tiết
└── gifs/                           # Thư mục chứa các tệp ảnh động GIF được xuất ra khi chạy với --record
```

---

## 2. Chi tiết 5 môi trường

| Môi trường | Loại bài toán | Số lượng Drone | Chiều Observation | Chiều Action | Mục tiêu chính |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`VelocityAviary`** | Single-Agent RL | 1 | 16 | 4 | Bám theo vector vận tốc mong muốn `[vx, vy, vz, yaw_rate]` |
| **`FlyThroughAviary`** | Single-Agent / Multi | 1 (hoặc N) | 18 | 4 | Bay qua tuần tự chuỗi Waypoint 3D trong không gian |
| **`FormationAviary`** | Cooperative Multi-Drone | 3 (hoặc N) | 54 (18 x 3) | 12 (4 x 3) | Giữ hình dạng đội hình (tam giác) trong khi tâm đội hình di chuyển |
| **`RaceAviary`** | Agile Racing RL | 1 (hoặc N) | 21 | 4 | Đua qua các cổng trên đường đua vòng tròn càng nhanh càng tốt |
| **`MultiAgentAviary`** | PettingZoo ParallelEnv | N (mỗi drone 1 agent) | 13 (mỗi agent) | 4 (mỗi agent) | MARL độc lập, phân công từng drone giữ độ cao lơ lửng khác nhau |

---

### Chi tiết kỹ thuật từng môi trường

#### 1. VelocityAviary
- **Observation Space (16 chiều)**:
  - `pos` (3): Tọa độ không gian [x, y, z]
  - `rpy` (3): Góc Euler roll, pitch, yaw
  - `vel` (3): Vận tốc tịnh tiến [vx, vy, vz]
  - `ang_v` (3): Vận tốc góc [wx, wy, wz]
  - `TARGET_VEL` (4): Vận tốc mục tiêu mong muốn [vx_des, vy_des, vz_des, yaw_rate_des]
- **Action Space (4 chiều)**: Tốc độ quay động cơ được chuẩn hóa trong khoảng `[-1, 1]`.
- **Hàm phần thưởng (Reward)**: Phạt sai số vận tốc `||vel - TARGET_VEL[:3]||` và sai số tốc độ quay yaw; cộng thưởng khi sai số < 0.05 m/s; phạt -100 nếu rơi hoặc lật nghiêng quá 90 độ.

#### 2. FlyThroughAviary
- **Observation Space (18 chiều)**: State cơ bản (12) + Tọa độ Waypoint tiếp theo (3) + Vector tương đối đến Waypoint `rel_wp = wp - pos` (3).
- **Hành trình**: Danh sách chuỗi tọa độ Waypoints trong không gian 3D.
- **Cơ chế nhận diện**: Khi khoảng cách giữa drone và waypoint < `WAYPOINT_RADIUS` (0.15m), hệ thống tự động nhảy sang Waypoint kế tiếp và cộng thưởng lớn **+10.0 điểm**.

#### 3. FormationAviary
- **Observation Space (18 x N chiều)**: Toàn bộ trạng thái của N drone cùng tọa độ mục tiêu đội hình tương ứng của từng drone.
- **Vector bù đắp (Offsets)**: Mỗi drone duy trì vị trí tương đối so với tâm hình học (ví dụ: các đỉnh của tam giác đều bán kính 0.3m).
- **Hàm phần thưởng (Reward)**: Vừa đánh giá khả năng bám theo tâm hành trình của từng cá nhân, vừa phạt nặng nếu khoảng cách thực tế giữa các drone bị lệch khỏi khoảng cách thiết kế lý tưởng (giúp ngăn va chạm và giữ cự ly đội hình).

#### 4. RaceAviary
- **Observation Space (21 chiều)**: State cơ bản (12) + Cổng tiếp theo `next_gate` (3) + Vector tương đối đến cổng (3) + Cổng kế tiếp sau đó `gate_after` (3) để drone có thể "nhìn trước" cua rẽ.
- **Hàm phần thưởng (Reward)**: Thưởng lớn **+20.0 điểm** khi vượt qua mỗi cổng + Thưởng tốc độ bay `2.0 * ||vel||` khuyến khích drone bay nhanh qua cổng mà không bị chững lại.

#### 5. MultiAgentAviary
- **Kiến trúc**: Giao diện PettingZoo `ParallelEnv` (`env.possible_agents = ['drone0', 'drone1', 'drone2']`).
- **Dict I/O**:
  - `actions = {"drone0": act0, "drone1": act1, "drone2": act2}`
  - `observations, rewards, terminations, truncations, infos = env.step(actions)`
- **Ứng dụng**: Tương thích trực tiếp với các framework Reinforcement Learning đa tác tử như Tianshou, CleanRL, Ray RLlib, hoặc Stable-Baselines3 (thông qua SuperSuit wrapper).

---

## 3. Hướng dẫn chạy thử nghiệm

Di chuyển terminal vào thư mục test:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos
```

### Cách 1: Chạy có giao diện 3D tương tác trực tiếp của MuJoCo (`--gui`)

Trong chế độ này, cửa sổ 3D của MuJoCo sẽ mở lên:
- **Chuột trái**: Xoay góc nhìn camera
- **Chuột phải**: Thu phóng (Zoom in / Zoom out)
- **Chuột giữa / Con lăn**: Di chuyển khung hình (Pan)
- **Phím Space**: Tạm dừng / Tiếp tục mô phỏng

```powershell
# 1. Xem VelocityAviary bám vận tốc
python 01_test_velocity_aviary.py --gui

# 2. Xem FlyThroughAviary bay lần lượt qua 5 Waypoints
python 02_test_fly_through_aviary.py --gui

# 3. Xem FormationAviary 3 drone bay đội hình tam giác
python 03_test_formation_aviary.py --gui

# 4. Xem RaceAviary drone đua qua các cổng
python 04_test_race_aviary.py --gui

# 5. Xem MultiAgentAviary 3 tác tử PettingZoo giữ các độ cao khác nhau
python 05_test_multi_agent_aviary.py --gui
```

---

### Cách 2: Ghi hình và lưu ra file ảnh động GIF (`--record`)

Khi chạy với cờ `--record`, hệ thống sẽ sử dụng offscreen renderer để render từng khung hình và tự động xuất file `.gif` vào thư mục `gifs/`:

```powershell
# Xuất GIF cho từng môi trường
python 01_test_velocity_aviary.py --record
python 02_test_fly_through_aviary.py --record
python 03_test_formation_aviary.py --record
python 04_test_race_aviary.py --record
python 05_test_multi_agent_aviary.py --record
```

---

### Cách 3: Chạy tổng hợp tất cả qua `run_all.py`

```powershell
# Chạy kiểm tra nhanh 5 môi trường (chế độ telemetry console)
python run_all.py

# Chạy và ghi lại toàn bộ GIF cho cả 5 môi trường
python run_all.py --record

# Chạy một môi trường cụ thể với GUI
python run_all.py --env fly_through --gui
```

---

### Chế độ kiểm tra ngẫu nhiên (`--random`)

Mặc định các kịch bản sử dụng bộ điều khiển PID khép kín (Autonomous PID Tracking) để drone bay ổn định và hoàn thành mục tiêu nhiệm vụ rõ ràng. Nếu bạn muốn kiểm tra hành vi lấy mẫu ngẫu nhiên của Reinforcement Learning Agent trước khi huấn luyện (RL random baseline), hãy thêm cờ `--random`:

```powershell
python 01_test_velocity_aviary.py --random
```
