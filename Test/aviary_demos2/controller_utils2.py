"""Tiện ích hỗ trợ kiểm tra môi trường Chướng ngại vật (Obstacles Environments).

Cung cấp:
- Bộ chuyển đổi RPM sang không gian hành động chuẩn hóa [-1, 1]
- Bộ phân tích tham số CLI (GUI, Record GIF, số vật cản, seed, số bước)
- Hàm lưu ảnh động GIF chất lượng cao
"""

import argparse
import os
import numpy as np
from PIL import Image


def rpm_to_normalized_action(rpm: np.ndarray, env) -> np.ndarray:
    """Chuyển đổi RPM thực tế từ bộ điều khiển PID sang action chuẩn hóa [-1, 1]."""
    rpm = np.clip(rpm, 0.0, env.MAX_RPM)
    act = np.where(
        rpm <= env.HOVER_RPM,
        (rpm / env.HOVER_RPM) - 1.0,
        (rpm - env.HOVER_RPM) / (env.MAX_RPM - env.HOVER_RPM),
    )
    return np.clip(act, -1.0, 1.0)


def save_gif(frames: list, output_path: str, fps: int = 20) -> str:
    """Lưu chuỗi khung hình RGB thành tệp ảnh động GIF."""
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


def parse_obstacle_args(theme_name: str = "OBSTACLE THEME") -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho các bài test chướng ngại vật."""
    parser = argparse.ArgumentParser(description=f"Test Môi trường Chướng ngại vật: {theme_name}")
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
        "--num-obstacles",
        type=int,
        default=25,
        help="Số lượng chướng ngại vật tạo ngẫu nhiên trong đấu trường",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên để tái lặp vị trí chướng ngại vật (mặc định: 42)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="track",
        choices=["track", "fixed", "fpv"],
        help="Chế độ camera: 'track' (bám theo drone), 'fixed' (toàn cảnh), 'fpv' (góc nhìn người lái)",
    )
    return parser.parse_args()
