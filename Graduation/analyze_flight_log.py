import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze(file_path: str, window_size: int = 50, save_fig: bool = True, show_fig: bool = True):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file log: {file_path}")
        print("💡 Hãy chạy huấn luyện PPO trước để sinh file nhật ký bay CSV.")
        return

    print(f"📂 Đang phân tích file nhật ký: {file_path}")
    df = pd.read_csv(file_path)

    if df.empty or 'Eposide' not in df.columns:
        print("⚠️ File log trống hoặc không có cột 'Eposide'!")
        return

    # Lấy thông số ở dòng cuối cùng của từng Episode (chuẩn tương tự ana1.py của D3QN)
    episodes_summary = df.groupby('Eposide').last().reset_index()
    total_episodes = len(episodes_summary)

    if total_episodes == 0:
        print("⚠️ Chưa có episode nào hoàn thành trong file log.")
        return

    overall_win_rate = episodes_summary['Win'].mean() * 100
    overall_collision_rate = episodes_summary['Collision'].mean() * 100
    overall_timeout_rate = episodes_summary['Over_Step'].mean() * 100 if 'Over_Step' in episodes_summary.columns else 0.0
    overall_overmap_rate = episodes_summary['Over_Map'].mean() * 100 if 'Over_Map' in episodes_summary.columns else 0.0
    avg_reward = episodes_summary['AccumReward'].mean()

    print("\n" + "=" * 55)
    print("      📊 BÁO CÁO TỔNG KẾT NHẬT KÝ BAY DRONE (VER2)")
    print("=" * 55)
    print(f"📌 Tổng số tập (Episodes)       : {total_episodes:,}")
    print(f"🏆 Tỷ lệ tới đích (Win Rate)    : {overall_win_rate:.2f}%")
    print(f"💥 Tỷ lệ đâm đụng (Collision)  : {overall_collision_rate:.2f}%")
    print(f"⏳ Tỷ lệ hết giờ (Timeout)      : {overall_timeout_rate:.2f}%")
    print(f"🚧 Tỷ lệ văng map (Over Map)    : {overall_overmap_rate:.2f}%")
    print(f"🎯 Điểm thưởng trung bình (Avg) : {avg_reward:.2f}")

    if 'Level' in episodes_summary.columns:
        print("-" * 55)
        print("📈 Thống kê theo từng Cấp độ Curriculum:")
        for lvl, group in episodes_summary.groupby('Level'):
            lvl_win = group['Win'].mean() * 100
            lvl_col = group['Collision'].mean() * 100
            lvl_rew = group['AccumReward'].mean()
            print(f"   Level {lvl} ({len(group)} tập): Win={lvl_win:.1f}% | Crash={lvl_col:.1f}% | AvgRew={lvl_rew:.1f}")

    print("=" * 55)
    print("\n--- 5 Tập gần nhất ---")
    cols_to_show = [c for c in ['Eposide', 'Level', 'AccumReward', 'Win', 'Collision', 'Over_Step'] if c in episodes_summary.columns]
    print(episodes_summary[cols_to_show].tail(5).to_string(index=False))

    # Tính toán đường trung bình trượt (Moving Average)
    episodes_summary['Moving_Reward'] = episodes_summary['AccumReward'].rolling(window=window_size, min_periods=1).mean()
    episodes_summary['Moving_Win_Rate'] = episodes_summary['Win'].rolling(window=window_size, min_periods=1).mean() * 100
    episodes_summary['Moving_Collision'] = episodes_summary['Collision'].rolling(window=window_size, min_periods=1).mean() * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Đồ thị 1: Điểm thưởng tích lũy (Reward)
    axes[0].plot(episodes_summary['Eposide'], episodes_summary['AccumReward'], alpha=0.25, color='gray', label='Điểm từng tập')
    axes[0].plot(episodes_summary['Eposide'], episodes_summary['Moving_Reward'], color='blue', linewidth=2, label=f'Trung bình trượt ({window_size} tập)')
    axes[0].set_title('Biểu đồ Điểm thưởng (Accumulated Reward) qua các hiệp')
    axes[0].set_ylabel('Điểm thưởng')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left')

    # Đồ thị 2: Tỷ lệ Win Rate & Collision Rate
    axes[1].plot(episodes_summary['Eposide'], episodes_summary['Moving_Win_Rate'], color='green', linewidth=2, label=f'Win Rate trượt ({window_size} tập)')
    axes[1].plot(episodes_summary['Eposide'], episodes_summary['Moving_Collision'], color='red', linewidth=1.5, linestyle='--', label=f'Collision Rate trượt')
    axes[1].set_title('Biểu đồ Tỷ lệ chiến thắng (Win Rate) và Va chạm (Collision)')
    axes[1].set_ylabel('Tỷ lệ (%)')
    axes[1].set_ylim(-2, 105)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper left')

    # Đồ thị 3: Tiến trình Cấp độ Curriculum
    if 'Level' in episodes_summary.columns:
        axes[2].step(episodes_summary['Eposide'], episodes_summary['Level'], color='purple', linewidth=2, where='post', label='Cấp độ Curriculum')
        axes[2].set_title('Tiến trình Cấp độ Curriculum (Level 0 -> 3)')
        axes[2].set_ylabel('Cấp độ')
        axes[2].set_yticks([0, 1, 2, 3])
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].legend(loc='upper left')

    axes[2].set_xlabel('Số tập (Episode)')
    plt.tight_layout()

    if save_fig:
        out_img = os.path.splitext(file_path)[0] + "_chart.png"
        plt.savefig(out_img, dpi=150)
        print(f"\n🖼️ Đã lưu biểu đồ phân tích tại: {out_img}")

    if show_fig:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Phân tích file nhật ký bay Drone PPO (tương tự ana1.py)")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn file CSV cần phân tích")
    parser.add_argument("--window", type=int, default=50, help="Kích thước cửa sổ trượt (Mặc định: 50)")
    parser.add_argument("--no-show", action="store_true", help="Không mở cửa sổ pop-up hiển thị biểu đồ")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.file is None:
        default_file = os.path.join(current_dir, "logs", "drone_flight_log_parallel_ver4.csv")
        if not os.path.exists(default_file):
            alt_file = os.path.join(current_dir, "logs", "drone_flight_log_ver4.csv")
            args.file = alt_file if os.path.exists(alt_file) else default_file
        else:
            args.file = default_file

    analyze(args.file, window_size=args.window, save_fig=True, show_fig=not args.no_show)

if __name__ == "__main__":
    main()
