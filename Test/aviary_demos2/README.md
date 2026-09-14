# Hướng dẫn Kiểm tra Môi trường Chướng ngại vật (Obstacles) trong MuJoCo-Drones

Thư mục `aviary_demos2` chứa toàn bộ mã nguồn kiểm tra, trực quan hóa 3D và kết xuất ảnh động GIF cho hệ thống **Sinh chướng ngại vật ngẫu nhiên theo thủ tục (Procedural Obstacle Generation)** với 5 chủ đề chuyên biệt:

1. **`FOREST`** (Rừng cây: Thân gỗ trụ + Tán lá xanh hình cầu)
2. **`URBAN`** (Đô thị / Hẻm nhà cao tầng: Các khối nhà hộp bê tông và đèn cảnh báo nóc)
3. **`INDOOR`** (Trong nhà / Phòng kín: 4 bức tường xung quanh, trần nhà, bàn ghế, cột trụ)
4. **`RANDOM`** (Ngẫu nhiên hỗn hợp: Khối cầu, trụ, hộp xoay tự do và kích thước đa dạng)
5. **`GATES`** (Cổng đua liên hoàn: Chuỗi khung cổng FPV với vùng tâm portal phát sáng)

---

## 1. Cấu trúc thư mục

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos2\
├── controller_utils2.py          # Tiện ích: chuyển đổi RPM sang Action [-1, 1], CLI, xuất GIF
├── 01_test_obstacle_forest.py    # Test chủ đề FOREST (Rừng cây)
├── 02_test_obstacle_urban.py     # Test chủ đề URBAN (Đô thị)
├── 03_test_obstacle_indoor.py    # Test chủ đề INDOOR (Trong nhà / Phòng kín)
├── 04_test_obstacle_random.py    # Test chủ đề RANDOM (Ngẫu nhiên hỗn hợp)
├── 05_test_obstacle_gates.py     # Test chủ đề GATES (Cổng đua liên hoàn)
├── run_all_obstacles.py          # Script tổng hợp chạy cả 5 hoặc từng chủ đề
├── README.md                     # Tài liệu hướng dẫn chi tiết
└── gifs/                         # Thư mục lưu các tệp ảnh động GIF của từng chủ đề
    ├── obstacle_forest.gif
    ├── obstacle_urban.gif
    ├── obstacle_indoor.gif
    ├── obstacle_random.gif
    └── obstacle_gates.gif
```

---

## 2. Đặc điểm kỹ thuật của 5 chủ đề

| Chủ đề | Hình học chính | Màu sắc & Vật liệu | Đặc điểm địa hình & Thử thách |
| :--- | :--- | :--- | :--- |
| **`FOREST`** | Thân cây `cylinder`, Tán lá `sphere` | Thân nâu gỗ, tán lá xanh tự nhiên | Mật độ dày đặc, thử thách bay luồn lách giữa các thân cây và dưới tán lá. |
| **`URBAN`** | Tòa nhà `box`, Đèn nóc `sphere` | Xám bê tông, đá, kính, đèn đỏ nóc | Các khe hẹp đô thị (Urban Canyon), bề mặt phẳng dựng đứng, đường phố ngã tư. |
| **`INDOOR`** | 4 Tường `box`, Trần `box`, Bàn, Cột | Tường trắng ngà, gỗ nâu, cột xám | Không gian kín có trần hạn chế độ cao tối đa, hiệu ứng dội luồng khí cánh quạt. |
| **`RANDOM`** | Hỗn hợp `box`, `cylinder`, `sphere` | Đa sắc ngẫu nhiên, xoay góc Euler | Môi trường bừa bộn không theo quy luật, kiểm thử độ khái quát hóa của mô hình RL. |
| **`GATES`** | 2 Cột trụ `box/cylinder`, Xà ngang, Portal | Cam neon, vàng phản quang, tâm xanh ngọc | Chuỗi cổng hẹp nối tiếp nhau, kiểm tra kỹ năng bay chính xác qua tâm cửa sổ. |

---

## 3. Hướng dẫn chạy thử nghiệm

Mở terminal và di chuyển vào thư mục:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos2
```

### Chế độ xem trực tiếp 3D MuJoCo Viewer (`--gui`)

```powershell
# 1. Xem rừng cây FOREST
python 01_test_obstacle_forest.py --gui

# 2. Xem đô thị URBAN
python 02_test_obstacle_urban.py --gui

# 3. Xem phòng kín INDOOR
python 03_test_obstacle_indoor.py --gui

# 4. Xem vật cản ngẫu nhiên RANDOM
python 04_test_obstacle_random.py --gui

# 5. Xem chuỗi cổng đua GATES
python 05_test_obstacle_gates.py --gui
```

### Chế độ ghi hình GIF tự động (`--record`)

```powershell
python 01_test_obstacle_forest.py --record
python 02_test_obstacle_urban.py --record
python 03_test_obstacle_indoor.py --record
python 04_test_obstacle_random.py --record
python 05_test_obstacle_gates.py --record
```

### Chạy tổng hợp qua `run_all_obstacles.py`

```powershell
# Chạy kiểm tra console nhanh cho cả 5 chủ đề
python run_all_obstacles.py

# Xuất lại toàn bộ GIF cho cả 5 chủ đề
python run_all_obstacles.py --record

# Chạy một chủ đề bất kỳ với GUI
python run_all_obstacles.py --theme forest --gui
```

### Tùy chỉnh tham số mô phỏng
- `--num-obstacles <N>`: Thay đổi số lượng vật cản sinh ra (ví dụ: `--num-obstacles 40` để tạo rừng cây rậm rạp).
- `--seed <S>`: Đổi seed ngẫu nhiên để sinh ra các bản đồ địa hình mới lạ hoàn toàn khác nhau.
- `--steps <K>`: Số bước chạy mô phỏng.
