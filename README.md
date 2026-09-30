# HỆ THỐNG ĐIỀU KHIỂN QUADROTOR DRONE TỰ HÀNH NÉ VẬT CẢN 3D BẰNG HỌC TĂNG CƯỜNG (PPO) & CURRICULUM LEARNING

> **Khóa Luận Tốt Nghiệp:** Điều khiển Quadrotor Drone Tự hành Tránh Vật cản 3D sử dụng Học tăng cường PPO kết hợp Học chương trình (Curriculum Learning) và Hợp nhất Cảm biến Đa phương thức (FPV Camera + Kinematics) trong môi trường vật lý MuJoCo.

---

## 📑 BẢNG MỤC LỤC
1. [Cấu Trúc Dự Án & Công Dụng Từng File](#1-cấu-trúc-dự-án--công-dụng-từng-file)
2. [Giải Thuật Trường Thế Nhân Tạo (APF Hint)](#2-giải-thuật-trường-thế-nhân-tạo-apf-hint)
3. [Cơ Chế Học Chương Trình (Curriculum Learning)](#3-cơ-chế-học-chương-trình-curriculum-learning)
4. [Chi Tiết 7 Cấp Độ & Thiết Kế Độ Khó](#4-chi-tiết-7-cấp-độ--thiết-kế-độ-khó)
5. [Hệ Thống Hàm Thưởng 8 Thành Phần (Reward Function)](#5-hệ-thống-hàm-thưởng-8-thành-phần-reward-function)
6. [Điều Kiện Dừng Episode: Terminated & Truncated](#6-điều-kiện-dừng-episode-terminated--truncated)
7. [Hướng Dẫn Huấn Luyện & Kiểm Thử](#7-hướng-dẫn-huấn-luyện--kiểm-thử)
8. [Các Đề Xuất Nâng Cao Giá Trị Học Thuật Cho Khóa Luận](#8-các-đề-xuất-nâng-cao-giá-trị-học-thuật-cho-khóa-luận)

---

## 1. Cấu Trúc Dự Án & Công Dụng Từng File

### 1.1. Sơ đồ cây thư mục

```
c:\KLTN\MuJoCo-Drones\
│
├── README.md                           # TÀI LIỆU HƯỚNG DẪN & BÁO CÁO TỔNG QUAN HỆ THỐNG
├── pyproject.toml                      # Quản lý metadata và gói phụ thuộc (PEP 517/518/621)
├── setup.py                            # Tương thích ngược công cụ cài đặt pip
│
├── Graduation/                         # MODULE TRỌNG TÂM CỦA KHÓA LUẬN
│   ├── config.py                       # Siêu tham số & thông số cấu hình toàn hệ thống
│   ├── drone_ppo_curriculum_env.py     # Môi trường Gymnasium Ver4 (MuJoCo Physics + Spikes 3D)
│   ├── train_parallel_curriculum.py    # Huấn luyện đa tiến trình song song (SubprocVecEnv)
│   ├── train_ppo_curriculum.py         # Huấn luyện đơn luồng (Single-worker debugging)
│   ├── eval_model.py                   # Đánh giá trực quan 3D + Bảng Telemetry thời gian thực
│   ├── test_reliance.py                # Bài test chẩn đoán thực nghiệm mức độ ỷ lại (Ablation Study)
│   ├── flight_logger_callback.py       # Callback ghi lại nhật ký chuyến bay ra file .csv
│   ├── analyze_flight_log.py           # Phân tích dữ liệu & vẽ biểu đồ quỹ đạo không gian 3D
│   ├── test_env.py                     # Kiểm thử tính hợp lệ Stable-Baselines3 (check_env)
│   ├── test_gui.py                     # Kiểm thử cửa sổ MuJoCo 3D Viewer
│   ├── test_dual_window.py             # Kiểm thử hiển thị 2 cửa sổ song song (3D View + FPV Camera)
│   ├── models/                         # Thư mục chứa các checkpoint trọng số mạng (.zip)
│   └── logs/                           # Nhật ký huấn luyện TensorBoard & flight logs (.csv)
│
├── multi_drone_mujoco/                 # THƯ VIỆN NỀN TẢNG MÔ PHỎNG DRONE
│   ├── assets/cf2/                     # 7 file Mesh 3D (.obj) Crazyflie 2.X chính hãng & textures
│   ├── control/                        # Bộ điều khiển PID 48Hz tracking vị trí và góc nghiêng
│   └── envs/base_aviary.py             # Lớp nền tảng kết nối MuJoCo C-API với Gymnasium
│
├── Test/                               # KỊCH BẢN THỬ NGHIỆM CÁC DÒNG MÔI TRƯỜNG KHÁC
│   ├── aviary_demos/                   # Demo bay vận tốc, theo vết, bay đội hình
│   └── ...
│
├── OldCode/                            # LƯU TRỮ CÁC PHIÊN BẢN CŨ (Ver1, Ver2, Ver3)
└── demo_gifs/                          # Ảnh động (.gif) trực quan hóa kết quả bay
```

---

### 1.2. Công dụng chi tiết từng file trong `Graduation/`

| Tên File | Công Dụng Chi Tiết |
| :--- | :--- |
| **`config.py`** | Quản lý toàn bộ tham số môi trường: Tọa độ xuất phát `start_space`, dải mục tiêu đích theo 7 Level (`GOAL_Y_RANGES`), tọa độ 15 cột trụ (`OBSTACLE_POSITIONS`), thông số gai (`SPIKES_PER_PILLAR_MIN=10`, `SPIKES_PER_PILLAR_MAX=16`, `SPIKE_LENGTH_MAX=1.1m`), bán kính đích `GOAL_THRESHOLD=0.5m`, biên an toàn `COLLISION_MARGIN=0.05m`, góc lái camera `MAX_ALPHA_DEG=35°`, `MAX_BETA_DEG=30°`. |
| **`drone_ppo_curriculum_env.py`** | Trái tim của hệ sinh thái mô phỏng: Tự động phát sinh XML MuJoCo linh hoạt theo từng Level; mô hình hóa thân vỏ Crazyflie 4 cánh quạt quay; sinh ngẫu nhiên cành gai 3D; thu nhận ảnh FPV $64\times 64$; tính toán APF Hints; tính toán hàm thưởng 8 thành phần; giải quyết va chạm vật lý và biên giới hạn. |
| **`train_parallel_curriculum.py`** | Kịch bản huấn luyện chính: Tận dụng `SubprocVecEnv` chạy song song trên nhiều nhân CPU; tích hợp `GlobalCurriculumCallback` để tự động chuyển cấp (Level 0 $\rightarrow$ 6) dựa trên tỷ lệ thắng tích lũy; tự động lưu checkpoint vào `models/` và ghi TensorBoard logs vào `logs/`. |
| **`train_ppo_curriculum.py`** | Kịch bản huấn luyện đơn tiến trình, phục vụ kiểm thử nhanh thuật toán PPO khi debug trên máy đơn luồng. |
| **`eval_model.py`** | Bộ công cụ đánh giá & trực quan hóa: Tự động nạp model mới nhất; mở giao diện mô phỏng 3D chuyển động mượt; in bảng Telemetry tọa độ $[X, Y, Z]$ thời gian thực từng bước; phân định chính xác 4 kết quả kết thúc (Thắng, Đâm gai, Văng map, Hết giờ). |
| **`test_reliance.py`** | Bộ kiểm tra khoa học (**Ablation Study**): Đánh giá định lượng xem Drone có bị ỷ lại vào Hint hay không qua 4 kịch bản đối chứng: (1) Chuẩn mực, (2) Cắt Hint, (3) Bịt mắt camera FPV, (4) Đánh lừa bằng Hint ngược. |
| **`flight_logger_callback.py`** | Callback ghi chép từng bước bay: Lưu vết tọa độ $[X, Y, Z]$, vận tốc, góc nghiêng Euler, khoảng cách tới gai gần nhất và điểm thưởng tích lũy ra file CSV. |
| **`analyze_flight_log.py`** | Phân tích dữ liệu bay sau huấn luyện: Tự động vẽ đồ thị quỹ đạo 3D (3D Trajectory), đồ thị độ cao theo thời gian, phân bố vận tốc và phân tích độ mượt mà. |
| **`test_env.py`** | Script kiểm tra độ tương thích: Chạy kiểm thử tự động `check_env` từ thư viện Stable-Baselines3 và xác thực chu trình Reset/Step trên toàn bộ 7 Level. |
| **`test_gui.py`** | Kiểm thử cửa sổ MuJoCo 3D Viewer và bộ nạp mô hình quadrotor Crazyflie 2.X. |
| **`test_dual_window.py`** | Kiểm thử đồng bộ 2 cửa sổ: Cửa sổ 1 (Toàn cảnh 3D) và Cửa sổ 2 (Camera FPV gắn trên mũi Drone bằng Tkinter). |

---

## 2. Giải Thuật Trường Thế Nhân Tạo (APF Hint)

Để hỗ trợ mạng nơ-ron định hướng ban đầu mà không rơi vào các cực tiểu cục bộ (local minima), hàm `_compute_apf_hint()` tính toán **2 tín hiệu gợi ý chuẩn hóa trong $[-1.0, 1.0]$** được đính kèm vào 2 chiều cuối của vector trạng thái:

```
[Mục Tiêu Đích] ---> Lực hút (Attractive Force) ----\
                                                     +--> Vector Vận Tốc Mong Muốn --> [Chuyển sang Body Frame]
[Cột & Gai Nhọn] --> Né Hành Lang + Đẩy Khẩn Cấp --/                                            |
                                                                                                v
                                                                             alpha_hint in [-1, 1] (Bẻ lái ngang)
                                                                             beta_hint  in [-1, 1] (Nâng/chúc Z)
```

### 2.1. `alpha_hint` — Gợi Ý Góc Bẻ Lái Ngang (Steering / Yaw Hint)
Tương ứng dải góc bẻ lái ngang $[-35^\circ, +35^\circ]$ (khống chế trong tầm nhìn FPV):
1. **Lực hút mục tiêu (Attractive):** Tạo vector đơn vị $\vec{u}_{\text{goal}}$ kéo thẳng về vị trí đích trên mặt phẳng $XY$.
2. **Dự đoán né hành lang va chạm (Anticipatory Corridor Deflection):** Quét các cột nằm phía trước trong cự ly $0.15\text{m} \sim 3.5\text{m}$. Nếu cột xâm lấn vào hành lang an toàn $r_{\text{corridor}} = 0.85\text{m}$, thuật toán tính góc lệch dạt mép $\theta = \arctan(\frac{d_{\text{clearance}}}{\text{proj}})$ và dùng tích có hướng $(\vec{u}_{\text{goal}} \times \vec{v}_{\text{obs}})$ để chọn hướng rẽ thoát nhanh nhất.
3. **Lực đẩy khẩn cấp sát bề mặt (Emergency Surface Repulsion):** Khi cự ly tới thân cột hoặc ngọn gai $< 0.6\text{m}$, sinh thêm vector lực đẩy tỷ lệ nghịch với khoảng cách ($F_{\text{rep}} \propto \frac{0.6 - d}{0.6}$) để bẻ lái khẩn cấp.

### 2.2. `beta_hint` — Gợi Ý Nâng / Hạ Cao Độ (Pitch / Altitude Hint)
Tương ứng dải góc ngẩng/chúc $[-30^\circ, +30^\circ]$:
- Tính độ lệch cao độ giữa quả cầu đích ($Z_g = 1.5\text{m}$) và vị trí Drone hiện tại: $\Delta Z = Z_{\text{goal}} - Z_{\text{drone}}$.
- Góc ngẩng mong muốn: $\theta_{\text{pitch}} = \arctan\left(\frac{\Delta Z}{d_{\text{goal}}}\right)$.
- Nếu Drone bay tà tà sát sàn ($Z = 0.8\text{m}$), $\Delta Z = +0.7\text{m} > 0 \implies \beta_{\text{hint}} > 0$ nhắc Drone phải ngẩng đầu bay lên cao để chạm đúng tâm quả cầu đích.

---

## 3. Cơ Chế Học Chương Trình (Curriculum Learning)

### 3.1. Tại sao cần Curriculum Learning?
Huấn luyện máy bay tự hành né 15 cột gai nhọn trên quãng đường 21m ngay từ đầu là bài toán có không gian khám phá cực lớn. Xác suất ngẫu nhiên để một Drone bay thẳng 21m mà không đâm gai gần như bằng $0\%$. 
Curriculum Learning giải quyết vấn đề này bằng cách chia lộ trình thành 7 nấc thang từ dễ đến khó: từ tập cất cánh trống trơn đến luồn lách qua mê cung gai rậm rạp.

### 3.2. Thuật toán tự động chuyển cấp (`GlobalCurriculumCallback`)
- **Cửa sổ đánh giá (`window_size = 100`):** Theo dõi kết quả của 100 episode gần nhất được tổng hợp từ tất cả các CPU worker song song.
- **Thềm thăng cấp (`threshold_advance = 0.8`):** Khi tỷ lệ tới đích thành công $\ge 80\%$, hệ thống tự động tăng cấp độ môi trường (`current_level += 1`).
- **Thềm hạ cấp (`threshold_retreat = 0.2`):** Nếu lên cấp mới mà tỷ lệ thắng rớt xuống dưới $20\%$, hệ thống tự động lùi lại 1 cấp (`current_level -= 1`) để củng cố chính sách bay trước khi thử lại.

---

## 4. Chi Tiết 7 Cấp Độ & Thiết Kế Độ Khó

| Cấp Độ (Level) | Khoảng Đích $Y$ | Cự Ly Bay | Số Cột Trụ | Mật Độ Gai | Thách Thức Kỹ Thuật |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Level 0** | $[7.5, 8.5]$ | $\sim 3\text{m}$ | **0 cột** | 0 | Tập cất cánh, cân bằng PID, bay thẳng $Z \in [1.0, 1.5]\text{m}$. |
| **Level 1** | $[-0.5, 0.5]$ | $\sim 11\text{m}$ | **3 cột** | $10-16$ gai/cột | Quãng đường tăng gấp 3; làm quen với việc quan sát cột ở xa. |
| **Level 2** | $[-0.5, 0.5]$ | $\sim 11\text{m}$ | **6 cột** | $10-16$ gai/cột | Xuất hiện các hàng cột so le; bắt đầu phải lượn $S$-curve né gai. |
| **Level 3** | $[-5.5, -4.5]$ | $\sim 16\text{m}$ | **8 cột** | $10-16$ gai/cột | Tăng cự ly lên 16m; bổ sung thêm hàng cột trung gian. |
| **Level 4** | $[-5.5, -4.5]$ | $\sim 16\text{m}$ | **10 cột** | $10-16$ gai/cột | Mật độ cột dày đặc; khe hở giữa các gai bị thu hẹp đáng kể. |
| **Level 5** | $[-10.5, -9.5]$ | $\sim 21\text{m}$ | **12 cột** | $10-16$ gai/cột | Quãng đường tối đa 21m; kiểm tra độ ổn định và bền bỉ của chính sách. |
| **Level 6** | $[-10.5, -9.5]$ | $\sim 21\text{m}$ | **15 cột** | $10-16$ gai/cột | **Cấp độ tối thượng:** Bổ sung Cột 14 ($X=-2.0$) và Cột 15 ($X=+2.0$) tại $Y=-8.0\text{m}$ tạo thành **Cổng gai đôi (Choke Point Gate)** chốt chặn ngay trước vạch đích. |

---

## 5. Hệ Thống Hàm Thưởng 8 Thành Phần (Reward Function)

Hàm thưởng substep được tính toán tại mỗi chu kỳ PID con (~48Hz) nhằm đảm bảo phản hồi tức thời:

$$\mathcal{R}_{\text{total}} = R_{\text{progress}} + R_{\text{regress}} + R_{\text{milestone}} + R_{\text{heading}} + R_{\text{idle}} + R_{\text{proximity}} + R_{\text{altitude\_boundary}} + R_{\text{terminal}}$$

```
+---------------------------------------------------------------------------------------------------+
|  1. R_progress          = +1.0 * (best_dist - cur_dist)   [Thưởng phá kỷ lục tiếp cận đích]       |
|  2. R_regress           = -3.0 * (cur_dist - prev_dist)   [Phạt nặng gấp 3 lần nếu bay thụt lùi]  |
|  3. R_milestone         = +5.0 (Duy nhất 1 lần/cột)       [Thưởng lách qua cột ở cự ly an toàn]   |
|  4. R_heading           = -0.01 * |cos(theta)|             [Phạt bay lệch/ngược hướng đích]        |
|  5. R_idle              = -0.005                          [Phạt đứng lỳ khi đường trống]          |
|  6. R_proximity         = -0.005                          [Cảnh báo nguy cơ khi quá sát gai <0.1m]|
|  7. R_altitude_boundary = -0.01                           [Phạt tì đè biên độ cao Z <= 0.85m]     |
|  8. R_terminal          = +100.0 (Win) / -50.0 (Va chạm / Văng map / Timeout)                     |
+---------------------------------------------------------------------------------------------------+
```

### Chi tiết các thành phần:
1. **$R_{\text{progress}}$ (High-Water Mark):** Chỉ thưởng khi Drone phá vỡ khoảng cách kỷ lục gần đích nhất từng đạt được trong episode.
2. **$R_{\text{regress}}$ (Anti-Backtracking):** Nếu Drone quay đầu bay xa đích hơn bước trước đó, bị phạt gấp 3 lần lượng thưởng tiến độ để triệt tiêu hành vi bay lượn vòng câu giờ.
3. **$R_{\text{milestone}}$:** Thưởng $+5.0$ ngay khi tọa độ $Y_{\text{drone}} \le Y_{\text{obs}}$ với khoảng cách an toàn $0.1\text{m} \le d \le 0.6\text{m}$.
4. **$R_{\text{altitude\_boundary}}$ (Cơ chế chống bẫy 2D):** 
   - Nếu $Z \le 0.85\text{m}$ (sát sàn) hoặc $Z \ge 2.45\text{m}$ (sát trần), phạt $-0.01$ mỗi substep (tích lũy phạt tới $-1.5$ điểm mỗi action step).
   - Buộc Drone phải bay lơ lửng tự do trong vùng an toàn $Z \in [1.0, 2.0]\text{m}$.

---

## 6. Điều Kiện Dừng Episode: Terminated & Truncated

Tuân thủ nghiêm ngặt chuẩn Gymnasium API:

### 6.1. `Terminated` (Kết Thúc Do Đạt Trạng Thái Tận Cùng Môi Trường)
`terminated = True` khi một trong 3 điều kiện sau xảy ra:
1. **Chiến thắng (`win = True`):** 
   - Khoảng cách không gian 3D Euclidean từ Drone tới tâm quả cầu đích:
     $$d = \sqrt{(X - X_g)^2 + (Y - Y_g)^2 + (Z - Z_g)^2} < \text{GOAL\_THRESHOLD} = 0.5\text{m}$$
   - Đích đặt tại $Z_g = 1.5\text{m}$. Do đó Drone bay sát sàn ($Z = 0.8\text{m}$) có $\Delta Z = 0.7\text{m} > 0.5\text{m}$ **không thể thắng**, bắt buộc phải nâng độ cao $Z \in [1.0, 2.0]\text{m}$.
   - Thưởng: **$+100.0$**.
2. **Va chạm (`collision = True`):**
   - Bộ giải tiếp xúc MuJoCo ghi nhận va quệt vật lý giữa `drone0` với bất kỳ `obstacle`, `spike` hoặc `floor`.
   - Hoặc cự ly hình học tới mép gai vi phạm biên an toàn $\le \text{COLLISION\_MARGIN} = 0.05\text{m}$.
   - Phạt: **$-50.0$**.
3. **Vượt ngoài bản đồ (`over_map = True`):**
   - Văng ra hai bên sườn: $X < X_{\text{min\_limit}}$ hoặc $X > X_{\text{max\_limit}}$ (biên động bóp theo vị trí cột $\pm 1.5\text{m}$).
   - Bay giật lùi sau điểm xuất phát $> 1.0\text{m}$ hoặc bay quá đích mà chưa chạm vùng thắng.
   - Rơi xuống sàn $Z < 0.25\text{m}$ hoặc vọt quá trần cột $Z > 3.5\text{m}$.
   - Phạt: **$-50.0$**.

### 6.2. `Truncated` (Kết Thúc Do Hết Giới Hạn Bước / Timeout)
`truncated = True` khi số bước hành động PPO đạt trần:
$$\text{step\_count} \ge \text{MAX\_STEPS} = 140\text{ bước}$$
- Nếu hết 140 bước mà Drone vẫn chưa chạm đích, Drone bị phạt răn đe **$-50.0$** để triệt tiêu chiến thuật bay lơ lửng tại chỗ nhằm bảo toàn điểm.

---

## 7. Hướng Dẫn Huấn Luyện & Kiểm Thử

### 7.1. Cài đặt môi trường
```bash
# 1. Kích hoạt môi trường conda
conda activate drone_env

# 2. Cài đặt các gói phụ thuộc
pip install -e .
```

### 7.2. Huấn luyện mô hình PPO song song
```bash
cd Graduation
# Huấn luyện trên 4 CPU workers với 2 triệu bước
python train_parallel_curriculum.py --num-cpu 4 --timesteps 2000000
```

### 7.3. Đánh giá mô hình & xem trực quan 3D
```bash
# Xem thử ở Level 2 (6 cột gai)
python eval_model.py --level 2 --episodes 5

# Xem thử ở Level 6 (15 cột gai cao nhất)
python eval_model.py --level 6 --episodes 5

# Đánh giá không mở GUI (chỉ in bảng số liệu)
python eval_model.py --level 6 --episodes 10 --no-gui
```

### 7.4. Bài kiểm tra thực nghiệm chẩn đoán ỷ lại (Ablation Study)
```bash
# Chạy 10 lượt đối chứng cho 4 chế độ tại Level 2
python test_reliance.py --level 2 --episodes 10
```

---

## 8. Các Đề Xuất Nâng Cao Giá Trị Học Thuật Cho Khóa Luận

Dưới đây là 3 đề xuất định hướng nghiên cứu giúp báo cáo Luận văn tốt nghiệp đạt điểm xuất sắc:

1. **Trực quan hóa bản đồ chú ý thị giác (Attention / Grad-CAM trên ảnh FPV):**
   - Sử dụng Grad-CAM để trích xuất Feature Map từ tầng Conv cuối cùng của mạng NatureCNN.
   - Hiển thị bản đồ nhiệt (Heatmap) chồng lên ảnh FPV để chứng minh: Khi Drone tiếp cận gai nhọn, các vùng điểm ảnh chứa gai trên ảnh camera kích hoạt tín hiệu nơ-ron mạnh nhất.
2. **Kỹ thuật xáo trộn miền môi trường (Domain Randomization):**
   - Đưa biến thiên ngẫu nhiên vào cường độ ánh sáng, màu sắc gai xương rồng và thêm nhiễu gió Gaussian vào lực đẩy quadrotor.
   - Chứng minh mô hình có khả năng chống nhiễu vượt trội (Robustness), sẵn sàng chuyển đổi từ mô phỏng sang Drone thật ngoài đời (Sim-to-Real transfer).
3. **Phân tích bề mặt ranh giới Pareto (Pareto Frontier Analysis):**
   - Vẽ biểu đồ đối sánh giữa **Vận tốc bay trung bình** và **Khoảng cách an toàn tối thiểu tới gai**.
   - Chứng minh sự đánh đổi tối ưu giữa tốc độ hoàn thành quãng đường và độ an toàn của thuật toán PPO.
