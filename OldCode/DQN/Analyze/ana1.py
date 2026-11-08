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

file_name = find_csv("drone_flight_log_test_DQN.csv")
if not os.path.exists(file_name):
    file_name = find_csv("drone_flight_log_parallel_DQN.csv")

if not os.path.exists(file_name):
    print(f"❌ Không tìm thấy file flight log CSV cho Standard DQN!")
    sys.exit(1)

print(f"📊 Đang đọc dữ liệu từ: {file_name}")
df = pd.read_csv(file_name)

ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else 'Step')
if ep_col in df.columns and df[ep_col].duplicated().any():
    episodes_summary = df.groupby(ep_col).last().reset_index()
else:
    episodes_summary = df.copy()

total_episodes = len(episodes_summary)
overall_win_rate = episodes_summary['Win'].mean() * 100 if 'Win' in episodes_summary.columns else 0
overall_collision_rate = episodes_summary['Collision'].mean() * 100 if 'Collision' in episodes_summary.columns else 0
avg_reward = episodes_summary['AccumReward'].mean() if 'AccumReward' in episodes_summary.columns else 0

print(f"========== TỔNG KẾT HUẤN LUYỆN STANDARD DQN ==========")
print(f"Tổng số tập (Episodes)         : {total_episodes}")
print(f"Tỷ lệ chiến thắng (Win Rate)   : {overall_win_rate:.2f}%")
print(f"Tỷ lệ va chạm (Collision Rate) : {overall_collision_rate:.2f}%")
print(f"Điểm thưởng trung bình        : {avg_reward:.2f}")
print("=====================================================")

window_size = 50
episodes_summary['Moving_Reward'] = episodes_summary['AccumReward'].rolling(window=window_size, min_periods=1).mean()
episodes_summary['Moving_Win_Rate'] = episodes_summary['Win'].rolling(window=window_size, min_periods=1).mean() * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(episodes_summary[ep_col], episodes_summary['AccumReward'], alpha=0.3, color='gray', label='Điểm từng tập (Gốc)')
ax1.plot(episodes_summary[ep_col], episodes_summary['Moving_Reward'], color='blue', linewidth=2, label=f'Trung bình trượt ({window_size} tập)')
ax1.set_title('1. Biểu đồ Điểm thưởng (Accumulated Reward) qua các tập của Standard DQN (MuJoCo)')
ax1.set_ylabel('Điểm thưởng')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

ax2.plot(episodes_summary[ep_col], episodes_summary['Moving_Win_Rate'], color='green', linewidth=2, label=f'Win Rate trượt ({window_size} tập)')
ax2.set_title('2. Biểu đồ Tỷ lệ chiến thắng (Win Rate %) của Standard DQN (MuJoCo)')
ax2.set_xlabel('Tập (Episode)')
ax2.set_ylabel('Tỷ lệ thắng (%)')
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

out_path = os.path.join(base_dir, "ana1_dqn_reward_winrate.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị vào: {out_path}")
plt.show()
