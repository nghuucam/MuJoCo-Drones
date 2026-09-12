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

def get_stats(df, name):
    if df is None: return None
    n = len(df)
    w = df['Win'].sum()
    c = df['Collision'].sum()
    om = df['Over_Map'].sum()
    os_cnt = df['Over_Step'].sum()
    r_mean = df['AccumReward'].mean()
    r_min = df['AccumReward'].min()
    r_max = df['AccumReward'].max()
    return {
        'name': name,
        'total': n,
        'win': w, 'win_pct': w / n * 100,
        'collision': c, 'collision_pct': c / n * 100,
        'over_map': om, 'over_map_pct': om / n * 100,
        'over_step': os_cnt, 'over_step_pct': os_cnt / n * 100,
        'reward_mean': r_mean, 'reward_min': r_min, 'reward_max': r_max
    }

s_d3qn = get_stats(df_d3qn, 'D3QN')
s_dqn = get_stats(df_dqn, 'DQN')

print("=" * 75)
print("📋 BÁO CÁO THỐNG KÊ TOÀN DIỆN ĐỐI ĐẦU: D3QN VS DQN (MUJOCO)")
print("=" * 75)
print(f"{'Chỉ số đánh giá':<32} | {'D3QN':<18} | {'DQN':<18}")
print("-" * 75)
print(f"{'Tổng số tập thử nghiệm':<32} | {s_d3qn['total']:<18} | {s_dqn['total']:<18}")
print(f"{'Số lần Thắng (Win)':<32} | {s_d3qn['win']} ({s_d3qn['win_pct']:.2f}%)   | {s_dqn['win']} ({s_dqn['win_pct']:.2f}%)")
print(f"{'Số lần Va chạm (Collision)':<32} | {s_d3qn['collision']} ({s_d3qn['collision_pct']:.2f}%)   | {s_dqn['collision']} ({s_dqn['collision_pct']:.2f}%)")
print(f"{'Số lần Bay ra ngoài (Over_Map)':<32} | {s_d3qn['over_map']} ({s_d3qn['over_map_pct']:.2f}%)   | {s_dqn['over_map']} ({s_dqn['over_map_pct']:.2f}%)")
print(f"{'Số lần Hết bước (Over_Step)':<32} | {s_d3qn['over_step']} ({s_d3qn['over_step_pct']:.2f}%)   | {s_dqn['over_step']} ({s_dqn['over_step_pct']:.2f}%)")
print("-" * 75)
print(f"{'Reward Trung bình':<32} | {s_d3qn['reward_mean']:<18.2f} | {s_dqn['reward_mean']:<18.2f}")
print(f"{'Reward Cao nhất (Max)':<32} | {s_d3qn['reward_max']:<18.2f} | {s_dqn['reward_max']:<18.2f}")
print(f"{'Reward Thấp nhất (Min)':<32} | {s_d3qn['reward_min']:<18.2f} | {s_dqn['reward_min']:<18.2f}")
print("=" * 75)

# Grouped bar chart comparing outcomes %
categories = ['Thắng (Win)', 'Va chạm (Collision)', 'Ra ngoài (Over_Map)', 'Hết bước (Over_Step)']
d3qn_vals = [s_d3qn['win_pct'], s_d3qn['collision_pct'], s_d3qn['over_map_pct'], s_d3qn['over_step_pct']]
dqn_vals = [s_dqn['win_pct'], s_dqn['collision_pct'], s_dqn['over_map_pct'], s_dqn['over_step_pct']]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, d3qn_vals, width, label='D3QN', color='#2ecc71')
rects2 = ax.bar(x + width/2, dqn_vals, width, label='DQN', color='#e74c3c')

ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=11)
ax.set_title('So sánh Phân bố Kết cục 400 Tập Huấn luyện giữa D3QN và DQN', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

out_path = os.path.join(base_dir, "ana5_compare_outcomes_summary.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu biểu đồ phân bố kết cục vào: {out_path}")
