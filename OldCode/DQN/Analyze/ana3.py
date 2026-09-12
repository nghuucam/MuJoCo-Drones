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

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Epsilon vs Win Rate progression
ax1 = axes[0]
if df_d3qn is not None:
    w_d3qn = df_d3qn['Win'].rolling(window=30, min_periods=1).mean() * 100
    ax1.plot(df_d3qn['Episode'], w_d3qn, color='#27ae60', linewidth=2.5, label='D3QN Win Rate (SMA 30)')
if df_dqn is not None:
    w_dqn = df_dqn['Win'].rolling(window=30, min_periods=1).mean() * 100
    ax1.plot(df_dqn['Episode'], w_dqn, color='#c0392b', linewidth=2.5, linestyle='--', label='DQN Win Rate (SMA 30)')

ax1.set_ylabel('Win Rate (%)', color='#2c3e50', fontweight='bold')
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle='--', alpha=0.6)

if df_d3qn is not None and 'Epsilon' in df_d3qn.columns:
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_d3qn['Episode'], df_d3qn['Epsilon'], color='#8e44ad', linewidth=2, linestyle=':', label='Epsilon Decay')
    ax1_twin.set_ylabel('Tỷ lệ Khám phá (Epsilon)', color='#8e44ad', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#8e44ad')
    ax1_twin.set_ylim(-0.05, 1.05)

ax1.set_title('1. Tương quan giữa Giảm Epsilon và Tỷ lệ Chiến thắng (D3QN vs DQN)', fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
if df_d3qn is not None and 'Epsilon' in df_d3qn.columns:
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend(loc='upper left')

# Subplot 2: Phân bố Reward theo 2 giai đoạn (Epsilon > 0.3 vs Epsilon <= 0.3)
ax2 = axes[1]
data_to_plot = []
labels = []
colors = ['#a8e6cf', '#27ae60', '#ffaaa5', '#c0392b']

if df_d3qn is not None and 'Epsilon' in df_d3qn.columns:
    d3qn_explore = df_d3qn[df_d3qn['Epsilon'] > 0.3]['AccumReward'].values
    d3qn_exploit = df_d3qn[df_d3qn['Epsilon'] <= 0.3]['AccumReward'].values
    data_to_plot.extend([d3qn_explore, d3qn_exploit])
    labels.extend(['D3QN Thăm dò (ε>0.3)', 'D3QN Khai thác (ε≤0.3)'])

if df_dqn is not None and 'Epsilon' in df_dqn.columns:
    dqn_explore = df_dqn[df_dqn['Epsilon'] > 0.3]['AccumReward'].values
    dqn_exploit = df_dqn[df_dqn['Epsilon'] <= 0.3]['AccumReward'].values
    data_to_plot.extend([dqn_explore, dqn_exploit])
    labels.extend(['DQN Thăm dò (ε>0.3)', 'DQN Khai thác (ε≤0.3)'])

if data_to_plot:
    bplot = ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    for patch, color in zip(bplot['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
    ax2.set_title('2. Phân bố Điểm thưởng (Reward Distribution) theo Giai đoạn Học', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Điểm thưởng (Reward)')
    ax2.grid(True, linestyle='--', alpha=0.6)

out_path = os.path.join(base_dir, "ana3_compare_epsilon_reward.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị Epsilon & Reward vào: {out_path}")
