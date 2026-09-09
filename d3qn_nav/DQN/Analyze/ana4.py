import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

def find_csv(filename):
    candidates = [
        os.path.join(parent_dir, "Single", filename),
        os.path.join(parent_dir, "Parallel", filename),
        os.path.join(parent_dir, filename),
        os.path.join(base_dir, filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def plot_moving_average(window_size=50):
    csv_file = find_csv("drone_flight_log_test_DQN.csv")
    if not os.path.exists(csv_file):
        csv_file = find_csv("drone_flight_log_parallel_DQN.csv")

    if not os.path.exists(csv_file):
        print(f"❌ Không tìm thấy file flight log CSV cho Standard DQN!")
        return

    print(f"📊 Đang đọc dữ liệu từ {csv_file}...")
    df = pd.read_csv(csv_file)
    
    ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else 'Step')
    if ep_col in df.columns and df[ep_col].duplicated().any():
        episodes_data = df.groupby(ep_col).last().reset_index()
        episodes_data['Step'] = df.groupby(ep_col)['Step'].max().values if 'Step' in df.columns else 1
    else:
        episodes_data = df.copy()

    episodes_data['MA_Reward'] = episodes_data['AccumReward'].rolling(window=window_size, min_periods=1).mean()
    episodes_data['MA_Steps'] = episodes_data['Step'].rolling(window=window_size, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Tiến trình Huấn luyện Standard DQN MuJoCo (Trung bình trượt {window_size} tập)', fontsize=14, fontweight='bold')

    ax1.plot(episodes_data[ep_col], episodes_data['AccumReward'], color='lightgray', alpha=0.6, label='Điểm từng tập (Gốc)')
    ax1.plot(episodes_data[ep_col], episodes_data['MA_Reward'], color='blue', linewidth=2, label=f'Điểm trung bình ({window_size} tập)')
    ax1.set_ylabel('Điểm thưởng (Reward)')
    ax1.set_title('1. Sự tăng trưởng Điểm số qua các Tập')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(episodes_data[ep_col], episodes_data['Step'], color='mistyrose', alpha=0.6, label='Số bước từng tập')
    ax2.plot(episodes_data[ep_col], episodes_data['MA_Steps'], color='red', linewidth=2, label=f'Số bước trung bình ({window_size} tập)')
    ax2.set_xlabel('Tập (Episode)')
    ax2.set_ylabel('Số bước bay (Steps)')
    ax2.set_title('2. Khả năng sống sót & Tối ưu số bước bay')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(base_dir, "ana4_dqn_reward_steps.png")
    plt.savefig(out_path, dpi=300)
    print(f"🖼️ Đã lưu đồ thị Reward & Steps vào: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_moving_average(window_size=50)
