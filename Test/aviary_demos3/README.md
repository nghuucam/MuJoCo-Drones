# Hướng dẫn Kiểm tra Wrapper Nhiễu loạn Gió (WindWrapper) trong MuJoCo-Drones

Thư mục `aviary_demos3` chứa toàn bộ mã nguồn kiểm tra, đánh giá và mô phỏng thực tế cho **[`WindWrapper`](file:///c:/KLTN/MuJoCo-Drones/multi_drone_mujoco/wrappers/wind_wrapper.py)** – lớp bọc Gymnasium chuyên dụng để tiêm lực nhiễu loạn khí động học (Aerodynamic Wind Disturbances) trực tiếp vào thân drone trong quá trình tính toán vật lý của MuJoCo.

---

## 1. Cấu trúc thư mục

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos3\
├── controller_utils3.py         # Tiện ích: chuyển đổi RPM sang Action [-1, 1], cờ gió 3D, xuất GIF
├── 01_test_wind_constant.py     # Test mô hình gió thổi liên tục (CONSTANT)
├── 02_test_wind_gust.py         # Test mô hình gió giật ngẫu nhiên (GUST)
├── 03_test_wind_dryden.py       # Test mô hình nhiễu loạn khí quyển tiêu chuẩn Dryden (DRYDEN)
├── 04_test_wind_sinusoidal.py   # Test mô hình gió dao động điều hòa hình sin (SINUSOIDAL)
├── 05_test_wind_combined.py     # Test mô hình gió bão kết hợp cực hạn (COMBINED)
├── 06_compare_wind_vs_nowind.py # Đánh giá đối chuẩn: Không gió vs Có gió
├── run_all_wind.py              # Script tổng hợp chạy toàn bộ kịch bản
├── README.md                    # Tài liệu hướng dẫn chi tiết
└── gifs/                        # Thư mục lưu các file ảnh động GIF đã kết xuất
    ├── wind_constant.gif
    ├── wind_gust.gif
    ├── wind_dryden.gif
    ├── wind_sinusoidal.gif
    ├── wind_combined.gif
    └── compare_benchmark.gif
```

---

## 2. Nguyên lý vật lý của `WindWrapper`

`WindWrapper` tính toán lực cản khí động và lực gió tác động lên thân drone tại mỗi bước mô phỏng:

$$\mathbf{F}_{\text{wind}} = \frac{1}{2} \rho \, (C_d \cdot A) \, \|\mathbf{v}_{\text{rel}}\| \, \mathbf{v}_{\text{rel}}$$

Trong đó:
- $\rho = 1.225 \text{ kg/m}^3$: Khối lượng riêng của không khí.
- $\mathbf{v}_{\text{rel}} = \mathbf{w}_{\text{wind}} - \mathbf{v}_{\text{drone}}$: Vận tốc tương đối giữa luồng gió và drone.
- $C_d \cdot A$: Hệ số cản khí động học diện tích thân quadrotor.
- Lực $\mathbf{F}_{\text{wind}}$ được tiêm trực tiếp vào mảng gia lực ngoài của MuJoCo `data.xfrc_applied[body_id]`.

---

## 3. Chi tiết 5 mô hình gió của `WindModel`

| Mô hình gió | Tham số chính | Đặc điểm vật lý & Biểu hiện của Drone |
| :--- | :--- | :--- |
| **`CONSTANT`** | `constant_wind=[wx, wy, wz]` | Gió thổi liên tục theo một hướng cố định. Drone phải nghiêng cánh (Pitch/Roll tilt angle) liên tục để bù gió và giữ vị trí hover. |
| **`GUST`** | `gust_intensity`, `gust_probability`, `gust_duration_steps` | Gió giật bất ngờ theo xung lực ngẫu nhiên. Drone bị giật mạnh đột ngột, kiểm tra khả năng phục hồi cân bằng của bộ điều khiển. |
| **`DRYDEN`** | `turbulence_intensity`, `altitude`, `airspeed` | Mô hình nhiễu loạn khí quyển tiêu chuẩn hàng không quân sự (MIL-F-8785C). Gió liên tục biến thiên đa chiều theo các dải tần số thực tế, làm rung lắc drone. |
| **`SINUSOIDAL`** | `sinusoidal_amplitude`, `sinusoidal_period` | Gió đảo chiều tuần hoàn dạng sóng sin $F(t) = A \sin(2\pi t / T)$. Drone đu đưa qua lại nhịp nhàng theo chu kỳ gió. |
| **`COMBINED`** | Tổng hợp cả 3 thành phần trên | Thử thách cực hạn: Gió nền liên tục + Nhiễu loạn Dryden + Gió giật bất ngờ (mô phỏng bay ngoài trời bão gió). |

---

## 4. Hướng dẫn chạy thử nghiệm

Mở terminal và di chuyển vào thư mục:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos3
```

### Cách 1: Xem trực tiếp cửa sổ 3D tương tác của MuJoCo (`--gui`)
*(Cột cờ gió 3D Windsock ở góc sân và vạch đích màu xanh lá hiển thị trực tiếp)*:

```powershell
# 1. Xem gió thổi ngang liên tục (drone nghiêng cánh bù gió)
python 01_test_wind_constant.py --gui

# 2. Xem gió giật bất thình lình
python 02_test_wind_gust.py --gui

# 3. Xem nhiễu loạn khí quyển Dryden làm rung lắc drone
python 03_test_wind_dryden.py --gui

# 4. Xem gió đảo chiều điều hòa hình sin
python 04_test_wind_sinusoidal.py --gui

# 5. Xem gió bão kết hợp cực hạn
python 05_test_wind_combined.py --gui

# 6. Chạy bảng đối chuẩn so sánh (Benchmark)
python 06_compare_wind_vs_nowind.py
```

### Cách 2: Tùy chỉnh tham số gió
```powershell
# Tăng vận tốc gió liên tục lên 3.0 m/s:
python 01_test_wind_constant.py --gui --wind-speed 3.0

# Tăng cường độ nhiễu loạn Dryden lên cấp 2.0 (mạnh):
python 03_test_wind_dryden.py --gui --turbulence 2.0

# Tăng lực gió giật lên 0.02 N:
python 02_test_wind_gust.py --gui --gust-intensity 0.02
```

### Cách 3: Chạy tổng hợp qua `run_all_wind.py`
```powershell
# Chạy kiểm tra nhanh toàn bộ các mô hình
python run_all_wind.py

# Xuất lại toàn bộ ảnh GIF vào thư mục gifs/
python run_all_wind.py --record

# Mở GUI xem riêng một kịch bản
python run_all_wind.py --model constant --gui
```

---

## 5. Thử nghiệm Trực quan Lực gió Rõ Rệt (Visible Wind Demo)

Nếu bạn muốn thấy tận mắt gió thổi bay drone hoặc drone phải nghiêng góc lớn để chống gió:

```powershell
# Chế độ 1: Thả trôi (DRIFT) - Bước 0-70 lặng gió đứng yên, bước 70 trở đi gió 6 m/s thổi bay dạt > 4 mét!
python 07_test_wind_drift_visible.py --gui --mode drift

# Chế độ 2: Kháng gió (RESIST) - Drone nghiêng hẳn góc Pitch -15° để chống lại luồng gió 6 m/s
python 07_test_wind_drift_visible.py --gui --mode resist
*(Trong giao diện 3D có Mũi tên khổng lồ chỉ hướng gió màu cam và các dải luồng khí động học màu xanh cyan)*.

---

## 6. Biểu tượng Luồng Gió Xoáy 3D Khí Động Học (Wind Gust Swirls)

Mô phỏng chính xác biểu tượng đồ họa của gió (3 dòng khí uốn lượn với các vòng xoáy tròn 3D ở đầu mút):

```powershell
# Xem trực tiếp trong cửa sổ 3D MuJoCo:
python 08_test_wind_swirls_icon.py --gui

# Hoặc qua run_all_wind.py:
python run_all_wind.py --model swirls --gui
```
*(Các luồng xoáy gió 3D tự động xoay theo vector gió thực tế và bao bọc lấy quỹ đạo bay của Drone).*


