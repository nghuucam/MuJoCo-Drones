import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
oldcode_dir = os.path.dirname(parent_dir)

def find_file(rel_paths):
    for p in rel_paths:
        if os.path.exists(p):
            return p
    return None

d3qn_candidates = [
    os.path.join(oldcode_dir, "D3QN", "Parallel", "drone_flight_log_parallel_D3QN.csv"),
    os.path.join(oldcode_dir, "D3QN", "Single", "drone_flight_log_test_D3QN.csv"),
    os.path.join(parent_dir, "Parallel", "drone_flight_log_parallel_D3QN.csv"),
    os.path.join(parent_dir, "Single", "drone_flight_log_test_D3QN.csv")
]

dqn_candidates = [
    os.path.join(oldcode_dir, "DQN", "Parallel", "drone_flight_log_parallel_DQN.csv"),
    os.path.join(oldcode_dir, "DQN", "Single", "drone_flight_log_test_DQN.csv"),
    os.path.join(oldcode_dir, "DQN", "Parallel", "drone_flight_log_parallel.csv")
]

f_d3qn = find_file(d3qn_candidates)
f_dqn = find_file(dqn_candidates)

def load_data(filepath):
    if not filepath or not os.path.exists(filepath):
        return None, None
    df = pd.read_csv(filepath)
    ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else 'Step')
    if ep_col in df.columns and df[ep_col].duplicated().any():
        df = df.groupby(ep_col).last().reset_index()
    return df, ep_col

df_d3qn, ep_d3qn = load_data(f_d3qn)
df_dqn, ep_dqn = load_data(f_dqn)

if df_d3qn is None and df_dqn is None:
    print("❌ Không tìm thấy file log CSV nào!")
    sys.exit(1)

print("=" * 70)
print("📊 BẢNG SO SÁNH HIỆU NĂNG TỔNG THỂ (ANA1: REWARD & WIN RATE)")
print("=" * 70)
print(f"{'Chỉ số':<30} | {'D3QN':<16} | {'DQN':<16}")
print("-" * 70)

def get_stats(df):
    if df is None: return 0, 0.0, 0.0
    tot = len(df)
    w_rate = df['Win'].mean() * 100 if 'Win' in df.columns else 0.0
    r_avg = df['AccumReward'].mean() if 'AccumReward' in df.columns else 0.0
    return tot, w_rate, r_avg

n1, w1, r1 = get_stats(df_d3qn)
n2, w2, r2 = get_stats(df_dqn)

print(f"{'Tổng số tập (Episodes)':<30} | {n1:<16} | {n2:<16}")
print(f"{'Tỷ lệ thắng (Win Rate %)':<30} | {w1:<15.2f}% | {w2:<15.2f}%")
print(f"{'Reward trung bình':<30} | {r1:<16.2f} | {r2:<16.2f}")
print("=" * 70)

window_size = 30

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

if df_d3qn is not None:
    df_d3qn['MA_Reward'] = df_d3qn['AccumReward'].rolling(window=window_size, min_periods=1).mean()
    df_d3qn['MA_Win'] = df_d3qn['Win'].rolling(window=window_size, min_periods=1).mean() * 100
    ax1.plot(df_d3qn[ep_d3qn], df_d3qn['AccumReward'], color='#2ecc71', alpha=0.18)
    ax1.plot(df_d3qn[ep_d3qn], df_d3qn['MA_Reward'], color='#27ae60', linewidth=2.5, label=f'D3QN (SMA {window_size})')
    ax2.plot(df_d3qn[ep_d3qn], df_d3qn['MA_Win'], color='#27ae60', linewidth=2.5, label=f'D3QN (SMA {window_size})')

if df_dqn is not None:
    df_dqn['MA_Reward'] = df_dqn['AccumReward'].rolling(window=window_size, min_periods=1).mean()
    df_dqn['MA_Win'] = df_dqn['Win'].rolling(window=window_size, min_periods=1).mean() * 100
    ax1.plot(df_dqn[ep_dqn], df_dqn['AccumReward'], color='#e74c3c', alpha=0.18)
    ax1.plot(df_dqn[ep_dqn], df_dqn['MA_Reward'], color='#c0392b', linewidth=2.5, linestyle='--', label=f'DQN (SMA {window_size})')
    ax2.plot(df_dqn[ep_dqn], df_dqn['MA_Win'], color='#c0392b', linewidth=2.5, linestyle='--', label=f'DQN (SMA {window_size})')

ax1.set_title('1. So sánh Điểm thưởng Tích lũy (Accumulated Reward) giữa D3QN và DQN', fontsize=13, fontweight='bold')
ax1.set_ylabel('Điểm thưởng', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=10)

ax2.set_title('2. So sánh Tỷ lệ Chiến thắng (Win Rate %) giữa D3QN và DQN', fontsize=13, fontweight='bold')
ax2.set_xlabel('Tập (Episode)', fontsize=11)
ax2.set_ylabel('Tỷ lệ thắng (%)', fontsize=11)
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', fontsize=10)

out_path = os.path.join(base_dir, "ana1_compare_reward_winrate.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị so sánh vào: {out_path}")
