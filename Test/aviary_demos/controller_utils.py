"""Helper utilities for testing Aviary environments.

Provides:
- Conversion between controller RPM and normalized [-1, 1] action space
- CLI argument parsing for interactive GUI and video/GIF recording
- Frame rendering and GIF export
- Formatted console telemetry display
"""

import argparse
import os
import numpy as np
from PIL import Image


def rpm_to_normalized_action(rpm: np.ndarray, env) -> np.ndarray:
    """Convert raw motor RPM to normalized action in [-1, 1].

    Parameters
    ----------
    rpm : np.ndarray
        Motor RPM command array.
    env : BaseAviary
        Environment containing HOVER_RPM and MAX_RPM attributes.

    Returns
    -------
    np.ndarray
        Normalized action in range [-1.0, 1.0].
    """
    rpm = np.clip(rpm, 0.0, env.MAX_RPM)
    act = np.where(
        rpm <= env.HOVER_RPM,
        (rpm / env.HOVER_RPM) - 1.0,
        (rpm - env.HOVER_RPM) / (env.MAX_RPM - env.HOVER_RPM),
    )
    return np.clip(act, -1.0, 1.0)


def save_gif(frames: list, output_path: str, fps: int = 20) -> str:
    """Save a sequence of RGB numpy frames as an animated GIF.

    Parameters
    ----------
    frames : list of np.ndarray
        List of RGB image arrays (H, W, 3).
    output_path : str
        Target file path for the .gif file.
    fps : int, optional
        Frames per second, default is 20.

    Returns
    -------
    str
        Absolute path to saved GIF.
    """
    if not frames:
        print("[WARN] No frames to save.")
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
    print(f"[SUCCESS] Đã lưu ảnh động GIF: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


def parse_demo_args(description: str = "Test Aviary Environment") -> argparse.Namespace:
    """Parse standard CLI arguments for aviary environment tests."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Mở cửa sổ đồ họa 3D tương tác của MuJoCo (render_mode='human')",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Ghi hình và xuất file GIF hoạt họa vào thư mục gifs/",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Số bước mô phỏng thực hiện (mặc định: 300 bước, khoảng 6.25 giây với 48Hz)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sử dụng hành động ngẫu nhiên (action space sample) thay vì bộ điều khiển PID",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="track",
        choices=["track", "fixed", "fpv"],
        help="Chế độ camera render: 'track' (bám theo drone), 'fixed' (cố định), 'fpv' (góc nhìn thứ nhất)",
    )
    return parser.parse_args()
