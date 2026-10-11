import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(current_dir, "logs")


def plot_single_failed_episode(ep_data, save_path=None, show=False):
    """Vẽ quỹ đạo 3D và góc nhìn 2D Top-down (Mặt bằng XY) cho 1 episode thất bại."""
    ep_idx = ep_data.get("episode_index", 1)
    reason = ep_data.get("reason", "Thất bại")
    result_type = ep_data.get("result_type", "unknown")
    start = np.array(ep_data["start_pos"])
    goal = np.array(ep_data["goal_pos"])
    traj = ep_data["trajectory"]
    obstacles = ep_data.get("obstacles", [])
    cyl_rad = ep_data.get("cylinder_radius", 0.3)

    xs = [pt["x"] for pt in traj]
    ys = [pt["y"] for pt in traj]
    zs = [pt["z"] for pt in traj]
    steps = [pt["step"] for pt in traj]

    fig = plt.figure(figsize=(16, 7), dpi=120)
    fig.suptitle(f"QUỸ ĐẠO MÀN THUA - EPISODE #{ep_idx} (Level {ep_data.get('level', 6)})\n"
                 f"Kết quả: {reason} | Tổng bước: {ep_data.get('total_steps')} | Cách đích: {ep_data.get('final_distance', 0):.2f}m",
                 fontsize=14, fontweight="bold", color="darkred" if result_type == "collision" else "darkblue")

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_title("Góc nhìn 3D Không gian", fontsize=12, fontweight="bold")

    for obs in obstacles:
        ox, oy, oh = obs["x"], obs["y"], obs["height"]
        z_cyl = np.linspace(0, oh, 10)
        theta = np.linspace(0, 2 * np.pi, 16)
        theta_grid, z_grid = np.meshgrid(theta, z_cyl)
        x_grid = ox + cyl_rad * np.cos(theta_grid)
        y_grid = oy + cyl_rad * np.sin(theta_grid)
        ax3d.plot_surface(x_grid, y_grid, z_grid, color="gray", alpha=0.3, edgecolor="none")

        for sp in obs.get("spikes", []):
            sp_x = ox + sp["x"]
            sp_y = oy + sp["y"]
            sp_z = sp["z"]
            ax3d.plot([ox, sp_x], [oy, sp_y], [sp_z, sp_z], color="brown", linewidth=1.2, alpha=0.6)

    ax3d.plot(xs, ys, zs, color="royalblue", linewidth=2.5, label="Đường bay Drone", zorder=5)

    if len(xs) > 1:
        step_interval = max(1, len(xs) // 8)
        for i in range(0, len(xs), step_interval):
            ax3d.scatter(xs[i], ys[i], zs[i], color="blue", s=20)
            ax3d.text(xs[i], ys[i], zs[i] + 0.05, f"#{steps[i]}", fontsize=8, color="navy")

    ax3d.scatter(start[0], start[1], start[2], color="forestgreen", s=120, marker="o", label="Xuất phát (Start)", zorder=10)
    ax3d.scatter(goal[0], goal[1], goal[2], color="crimson", s=180, marker="*", label="Đích (Goal)", zorder=10)

    ax3d.scatter(xs[-1], ys[-1], zs[-1], color="red", s=250, marker="X", label=f"Vị trí Thua: {reason}", zorder=15)

    ax3d.set_xlabel("Trục X (m)")
    ax3d.set_ylabel("Trục Y (m)")
    ax3d.set_zlabel("Độ cao Z (m)")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.grid(True, linestyle="--", alpha=0.5)

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_title("Mặt bằng 2D nhìn từ trên xuống (Top-down XY)", fontsize=12, fontweight="bold")

    for i, obs in enumerate(obstacles):
        ox, oy = obs["x"], obs["y"]
        circle = plt.Circle((ox, oy), cyl_rad, color="dimgray", alpha=0.5, zorder=2)
        ax2d.add_patch(circle)
        
        for sp in obs.get("spikes", []):
            sp_x = ox + sp["x"]
            sp_y = oy + sp["y"]
            ax2d.plot([ox, sp_x], [oy, sp_y], color="darkred", linewidth=1.5, alpha=0.7, zorder=3)

    ax2d.plot(xs, ys, color="dodgerblue", linewidth=2.5, label="Quỹ đạo bay", zorder=4)

    if len(xs) > 1:
        step_interval = max(1, len(xs) // 10)
        for i in range(0, len(xs), step_interval):
            ax2d.scatter(xs[i], ys[i], color="blue", s=25, zorder=5)
            ax2d.annotate(f"#{steps[i]}", (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="darkblue")

    ax2d.scatter(start[0], start[1], color="forestgreen", s=140, marker="o", label="Xuất phát (Start)", zorder=10)
    ax2d.scatter(goal[0], goal[1], color="crimson", s=220, marker="*", label="Đích (Goal)", zorder=10)

    ax2d.scatter(xs[-1], ys[-1], color="red", s=300, marker="X", label=f"Điểm dừng ({reason})", zorder=15)
    ax2d.annotate(f"Dừng tại bước #{steps[-1]} (Z={zs[-1]:.2f}m)",
                  (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -12),
                  fontsize=9, fontweight="bold", color="red",
                  bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6))

    ax2d.set_xlabel("Trục X (m)")
    ax2d.set_ylabel("Trục Y (m)")
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.grid(True, linestyle=":", alpha=0.6)
    ax2d.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"   🖼️ Đã lưu hình vẽ: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Đọc log quỹ đạo và vẽ chi tiết các màn chơi THUA của Drone")
    parser.add_argument("--input", type=str, default=os.path.join(LOGS_DIR, "failed_trajectories.json"),
                        help="Đường dẫn file JSON chứa các màn thua")
    parser.add_argument("--output-dir", type=str, default=os.path.join(LOGS_DIR, "failed_plots"),
                        help="Thư mục xuất ảnh quỹ đạo")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ biểu đồ Matplotlib tương tác")
    args = parser.parse_args()

    print("=" * 80)
    print("📈 CÔNG CỤ VẼ QUỸ ĐẠO CÁC MÀN CHƠI THUA (FAILED TRAJECTORY PLOTTER)")
    print("=" * 80)
    print(f"📂 Đọc file JSON từ : {args.input}")
    print(f"📁 Xuất ảnh ra      : {args.output_dir}")

    if not os.path.exists(args.input):
        print(f"❌ KHÔNG TÌM THẤY FILE: {args.input}")
        print("💡 Hãy chạy 'python eval_model.py' trước để tạo ra các màn chơi thua và lưu log!")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("🎉 Không có màn chơi thua nào trong file JSON (Tất cả đều thắng)!")
        return

    print(f"🔍 Tìm thấy {len(data)} màn chơi thất bại cần vẽ.")
    print("-" * 80)

    for i, ep_data in enumerate(data):
        ep_idx = ep_data.get("episode_index", i + 1)
        res_type = ep_data.get("result_type", "fail")
        save_file = os.path.join(args.output_dir, f"failed_ep_{ep_idx:02d}_{res_type}.png")
        print(f"🎨 Đang vẽ Episode #{ep_idx} ({ep_data.get('reason')})...")
        plot_single_failed_episode(ep_data, save_path=save_file, show=args.show)

    print("=" * 80)
    print(f"✅ HOÀN TẤT VẼ {len(data)} MÀN THUA! Mời bạn mở thư mục xem ảnh:")
    print(f"   📂 {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
