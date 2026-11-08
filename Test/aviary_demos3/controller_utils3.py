"""Tiện ích hỗ trợ kiểm tra WindWrapper cho mô phỏng Drone trong MuJoCo.

Cung cấp:
- Chuyển đổi RPM sang Action [-1, 1]
- Bộ phân tích tham số CLI
- Hàm xuất ảnh GIF
- Cột chỉ báo hướng gió trực quan (Windsock Pole) trong không gian 3D
"""

import argparse
import os
import numpy as np
from PIL import Image


def rpm_to_normalized_action(rpm: np.ndarray, env) -> np.ndarray:
    """Chuyển đổi RPM từ bộ điều khiển PID sang action chuẩn hóa [-1, 1]."""
    # Lấy base_env nếu env được wrap bởi WindWrapper
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    rpm = np.clip(rpm, 0.0, base_env.MAX_RPM)
    act = np.where(
        rpm <= base_env.HOVER_RPM,
        (rpm / base_env.HOVER_RPM) - 1.0,
        (rpm - base_env.HOVER_RPM) / (base_env.MAX_RPM - base_env.HOVER_RPM),
    )
    return np.clip(act, -1.0, 1.0)


def save_gif(frames: list, output_path: str, fps: int = 20) -> str:
    """Lưu chuỗi khung hình RGB thành ảnh động GIF."""
    if not frames:
        print("[WARN] Không có khung hình để lưu.")
        return ""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"[SUCCESS] Đã lưu hoạt ảnh GIF: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


def generate_wind_swirl_geoms(name_prefix="wind_swirl", scale=0.32, rgba="0.2 0.85 1.0 0.85", thickness=0.0055, plane="xz") -> str:
    """Tạo các đoạn hình học (capsules) ghép thành biểu tượng cuộn xoáy gió 3D (Wind Gust Swirl)
    dựa trên biểu tượng khí động học tiêu chuẩn (3 dòng khí uốn lượn có vòng cuộn tròn ở đầu mút).
    """
    geoms = []

    def add_capsule(p1, p2, geom_id):
        p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
        mid = (p1 + p2) / 2.0
        diff = p2 - p1
        length = float(np.linalg.norm(diff))
        if length < 1e-4:
            return ""
        dz = diff / length
        return f'<geom name="{name_prefix}_{geom_id}" type="capsule" size="{thickness:.4f} {length/2:.4f}" pos="{mid[0]:.3f} {mid[1]:.3f} {mid[2]:.3f}" zaxis="{dz[0]:.3f} {dz[1]:.3f} {dz[2]:.3f}" rgba="{rgba}" contype="0" conaffinity="0"/>'

    n_seg = 14

    # 1. Đường luồng trên (Top stream with upward curl)
    h_top = 0.35 * scale
    R_top = 0.26 * scale
    thetas_top = np.linspace(-np.pi / 2, 1.35 * np.pi, n_seg)
    c_x_top, c_h_top = 0.0, h_top + R_top

    # 2. Đường luồng giữa (Middle stream with farther upward curl)
    h_mid = 0.0
    x_mid_end = 0.65 * scale
    R_mid = 0.22 * scale
    thetas_mid = np.linspace(-np.pi / 2, 1.35 * np.pi, n_seg)
    c_x_mid, c_h_mid = x_mid_end, h_mid + R_mid

    # 3. Đường luồng dưới (Bottom stream with downward curl)
    h_bot = -0.35 * scale
    R_bot = 0.26 * scale
    thetas_bot = np.linspace(np.pi / 2, -1.35 * np.pi, n_seg)
    c_x_bot, c_h_bot = 0.0, h_bot - R_bot

    def map_coords(x, h):
        if plane == "xz":
            return [x, 0.0, h]
        elif plane == "xy":
            return [x, h, 0.0]
        else:
            return [x, 0.707 * h, 0.707 * h]

    # Luồng trên & vòng cuộn
    geoms.append(add_capsule(map_coords(-1.1 * scale, h_top), map_coords(0.0, h_top), "top_line"))
    for i in range(n_seg - 1):
        p1 = map_coords(c_x_top + R_top * np.cos(thetas_top[i]), c_h_top + R_top * np.sin(thetas_top[i]))
        p2 = map_coords(c_x_top + R_top * np.cos(thetas_top[i + 1]), c_h_top + R_top * np.sin(thetas_top[i + 1]))
        geoms.append(add_capsule(p1, p2, f"top_curl_{i}"))

    # Luồng giữa & vòng cuộn
    geoms.append(add_capsule(map_coords(-1.3 * scale, h_mid), map_coords(x_mid_end, h_mid), "mid_line"))
    for i in range(n_seg - 1):
        p1 = map_coords(c_x_mid + R_mid * np.cos(thetas_mid[i]), c_h_mid + R_mid * np.sin(thetas_mid[i]))
        p2 = map_coords(c_x_mid + R_mid * np.cos(thetas_mid[i + 1]), c_h_mid + R_mid * np.sin(thetas_mid[i + 1]))
        geoms.append(add_capsule(p1, p2, f"mid_curl_{i}"))

    # Luồng dưới & vòng cuộn
    geoms.append(add_capsule(map_coords(-1.1 * scale, h_bot), map_coords(0.0, h_bot), "bot_line"))
    for i in range(n_seg - 1):
        p1 = map_coords(c_x_bot + R_bot * np.cos(thetas_bot[i]), c_h_bot + R_bot * np.sin(thetas_bot[i]))
        p2 = map_coords(c_x_bot + R_bot * np.cos(thetas_bot[i + 1]), c_h_bot + R_bot * np.sin(thetas_bot[i + 1]))
        geoms.append(add_capsule(p1, p2, f"bot_curl_{i}"))

    return "\n      ".join(filter(None, geoms))


def get_windsock_xml(wind_dir=np.array([1.0, 0.0, 0.0]), wind_speed=1.5) -> str:
    """Tạo đối tượng trực quan hóa gió 3D trong MuJoCo: Cột cờ gió, Mũi tên gió và Các cuộn xoáy gió khí quyển."""
    norm = np.linalg.norm(wind_dir[:2])
    yaw = np.degrees(np.arctan2(wind_dir[1], wind_dir[0])) if norm > 1e-3 else 0.0
    arrow_len = min(0.8, max(0.3, norm * 0.15))
    
    # 3 Cụm xoáy gió chuẩn hóa theo mẫu người dùng cung cấp (Wind Swirl Gusts)
    swirl_left = generate_wind_swirl_geoms("swirl_L", scale=0.30, rgba="0.2 0.85 1.0 0.85", thickness=0.0055, plane="xz")
    swirl_right = generate_wind_swirl_geoms("swirl_R", scale=0.30, rgba="0.2 0.85 1.0 0.85", thickness=0.0055, plane="xz")
    swirl_top = generate_wind_swirl_geoms("swirl_T", scale=0.24, rgba="0.3 0.90 1.0 0.75", thickness=0.0045, plane="xz")
    
    return f"""
    <!-- Mũi tên chỉ hướng gió 3D trên cao (Giant Wind Direction Arrow) -->
    <body name="wind_arrow" pos="0 0 2.2" euler="0 0 {yaw:.1f}">
      <!-- Thân mũi tên màu đỏ cam -->
      <geom name="arrow_shaft" type="cylinder" size="0.025 {arrow_len:.2f}" pos="0 0 0" euler="0 90 0" rgba="1.0 0.3 0.0 0.9" contype="0" conaffinity="0"/>
      <!-- Đầu mũi tên hình nón/trụ màu vàng sáng -->
      <geom name="arrow_head" type="cylinder" size="0.06 0.12" pos="{arrow_len + 0.1:.2f} 0 0" euler="0 90 0" rgba="1.0 0.85 0.0 0.95" contype="0" conaffinity="0"/>
    </body>

    <!-- CÁC BIỂU TƯỢNG CUỘN XOÁY GIÓ KHÍ QUYỂN 3D (WIND GUST SWIRLS) THEO HƯỚNG GIÓ -->
    <!-- Cụm gió 1: Bay ngang bên trái drone -->
    <body name="wind_swirl_left" pos="0.0 0.34 1.02" euler="0 0 {yaw:.1f}">
      {swirl_left}
    </body>
    <!-- Cụm gió 2: Bay ngang bên phải drone -->
    <body name="wind_swirl_right" pos="0.0 -0.34 1.02" euler="0 0 {yaw:.1f}">
      {swirl_right}
    </body>
    <!-- Cụm gió 3: Bay phía trên đỉnh drone -->
    <body name="wind_swirl_top" pos="0.12 0.0 1.30" euler="0 0 {yaw:.1f}">
      {swirl_top}
    </body>

    <!-- Các vệt luồng khí trực quan phụ (Wind Streamlines) -->
    <body name="streamline_1" pos="0 0.8 1.4" euler="0 0 {yaw:.1f}">
      <geom type="cylinder" size="0.008 1.2" euler="0 90 0" rgba="0.4 0.8 1.0 0.35" contype="0" conaffinity="0"/>
    </body>
    <body name="streamline_2" pos="0 -0.8 1.4" euler="0 0 {yaw:.1f}">
      <geom type="cylinder" size="0.008 1.2" euler="0 90 0" rgba="0.4 0.8 1.0 0.35" contype="0" conaffinity="0"/>
    </body>

    <!-- Cột cờ gió trực quan (Weather Station / Windsock) -->
    <body name="weather_station" pos="-1.5 1.5 0">
      <geom name="ws_pole" type="cylinder" size="0.02 0.75" pos="0 0 0.75" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
      <geom name="ws_top" type="sphere" size="0.04" pos="0 0 1.5" rgba="1.0 0.2 0.2 1" contype="0" conaffinity="0"/>
      <geom name="ws_sock" type="cylinder" size="0.035 0.18" pos="0.18 0 1.48" euler="0 90 {yaw:.1f}" rgba="1.0 0.4 0.0 0.9" contype="0" conaffinity="0"/>
    </body>

    <!-- 4 Trụ mốc ranh giới sân bay -->
    <body name="wind_corner_0" pos="2 2 0.4"><geom type="cylinder" size="0.03 0.4" rgba="0.3 0.7 0.9 0.7" contype="1" conaffinity="1"/></body>
    <body name="wind_corner_1" pos="-2 2 0.4"><geom type="cylinder" size="0.03 0.4" rgba="0.3 0.7 0.9 0.7" contype="1" conaffinity="1"/></body>
    <body name="wind_corner_2" pos="-2 -2 0.4"><geom type="cylinder" size="0.03 0.4" rgba="0.3 0.7 0.9 0.7" contype="1" conaffinity="1"/></body>
    <body name="wind_corner_3" pos="2 -2 0.4"><geom type="cylinder" size="0.03 0.4" rgba="0.3 0.7 0.9 0.7" contype="1" conaffinity="1"/></body>

    <!-- Vạch đích Hover trung tâm -->
    <body name="hover_target_marker" pos="0 0 1.0">
      <geom name="target_sphere" type="sphere" size="0.04" rgba="0.1 1.0 0.3 0.6" contype="0" conaffinity="0"/>
      <geom name="target_beacon" type="cylinder" size="0.005 0.5" pos="0 0 -0.5" rgba="0.1 1.0 0.3 0.2" contype="0" conaffinity="0"/>
    </body>
"""


def parse_wind_args(title: str = "Test WindWrapper") -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho kịch bản test gió."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Mở cửa sổ đồ họa 3D tương tác của MuJoCo (render_mode='human')",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Ghi hình và xuất file GIF vào thư mục gifs/",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Số bước mô phỏng (mặc định: 300 bước ~ 6.25 giây)",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=3.5,
        help="Vận tốc gió danh định (m/s, mặc định: 3.5 m/s)",
    )
    parser.add_argument(
        "--turbulence",
        type=float,
        default=1.8,
        help="Cường độ nhiễu loạn Dryden (1.0=nhẹ, 1.8=vừa, 2.5=mạnh)",
    )
    parser.add_argument(
        "--gust-intensity",
        type=float,
        default=0.045,
        help="Lực gió giật tối đa (Newton, mặc định: 0.045 N ~ 17% trọng lượng drone)",
    )
    parser.add_argument(
        "--soft-pid",
        action="store_true",
        help="Bật chế độ PID mềm dẻo (compliant) để drone bồng bềnh và lung lay tự nhiên theo gió",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="track",
        choices=["track", "fixed", "fpv"],
        help="Chế độ camera: 'track' (bám theo drone), 'fixed' (cố định), 'fpv' (góc nhìn buồng lái)",
    )
    return parser.parse_args()
