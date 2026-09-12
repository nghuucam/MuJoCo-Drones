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
    os.path.join(parent_dir, "Parallel", "drone_flight_log_parallel_D3QN.csv")
]

dqn_candidates = [
    os.path.join(oldcode_dir, "DQN", "Parallel", "drone_flight_log_parallel_DQN.csv"),
    os.path.join(parent_dir, "Parallel", "drone_flight_log_parallel_DQN.csv")
]

f_d3qn = find_file(d3qn_candidates)
f_dqn = find_file(dqn_candidates)

df_d3qn = pd.read_csv(f_d3qn) if f_d3qn else None
df_dqn = pd.read_csv(f_dqn) if f_dqn else None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
window_size = 30

# Subplot 1: Cumulative Wins
if df_d3qn is not None:
    cum_win_d3qn = df_d3qn['Win'].cumsum()
    ax1.plot(df_d3qn['Episode'], cum_win_d3qn, color='#27ae60', linewidth=2.8, label=f'D3QN (Tổng thắng: {df_d3qn["Win"].sum()})')

if df_dqn is not None:
    cum_win_dqn = df_dqn['Win'].cumsum()
    ax1.plot(df_dqn['Episode'], cum_win_dqn, color='#c0392b', linewidth=2.8, linestyle='--', label=f'DQN (Tổng thắng: {df_dqn["Win"].sum()})')

ax1.set_title('1. So sánh Số trận Thắng Tích lũy (Cumulative Wins) qua các Tập', fontsize=13, fontweight='bold')
ax1.set_ylabel('Tổng số tập thắng tích lũy', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=11)

# Subplot 2: Moving Collision Rate (%)
if df_d3qn is not None:
    ma_coll_d3qn = df_d3qn['Collision'].rolling(window=window_size, min_periods=1).mean() * 100
    ax2.plot(df_d3qn['Episode'], ma_coll_d3qn, color='#27ae60', linewidth=2.5, label=f'D3QN Collision (SMA {window_size})')

if df_dqn is not None:
    ma_coll_dqn = df_dqn['Collision'].rolling(window=window_size, min_periods=1).mean() * 100
    ax2.plot(df_dqn['Episode'], ma_coll_dqn, color='#c0392b', linewidth=2.5, linestyle='--', label=f'DQN Collision (SMA {window_size})')

ax2.set_title('2. So sánh Tỷ lệ Va chạm (Collision Rate %) giữa D3QN và DQN', fontsize=13, fontweight='bold')
ax2.set_xlabel('Tập (Episode)', fontsize=11)
ax2.set_ylabel('Tỷ lệ va chạm (%)', fontsize=11)
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=11)

out_path = os.path.join(base_dir, "ana4_compare_cumulative_wins_collisions.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị Tích lũy Thắng & Va chạm vào: {out_path}")
