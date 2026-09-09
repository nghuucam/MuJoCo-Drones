import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

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

file_name = find_csv("neutron_array_D3QN.csv")
if not os.path.exists(file_name):
    file_name = find_csv("neutron_array_parallel_D3QN.csv")

if not os.path.exists(file_name):
    print(f"❌ Không tìm thấy file Q-Value / Action log CSV cho D3QN!")
    sys.exit(1)

print(f"📊 Đang đọc dữ liệu Q-Values và Action từ: {file_name}")
df = pd.read_csv(file_name, comment='#')

ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else 'Step')
act_col = 'Action_Decide' if 'Action_Decide' in df.columns else ('Action' if 'Action' in df.columns else None)

def extract_max_q(action_string):
    if pd.isna(action_string):
        return np.nan
    clean_string = str(action_string).replace('float32', '').replace('float64', '')
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", clean_string)
    if not numbers:
        return np.nan
    return max([float(num) for num in numbers])

if 'List_Actions' in df.columns:
    df['Max_Q_Value'] = df['List_Actions'].apply(extract_max_q)
else:
    df['Max_Q_Value'] = np.nan

episode_stats = df.groupby(ep_col).agg(
    Epsilon=('Epsilon', 'last') if 'Epsilon' in df.columns else (ep_col, 'count'),
    Avg_Max_Q=('Max_Q_Value', 'mean')
).reset_index()

episode_stats['Moving_Q'] = episode_stats['Avg_Max_Q'].rolling(window=20, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 1. Biểu đồ Q-Value & Epsilon
ax1 = axes[0]
if not episode_stats['Avg_Max_Q'].dropna().empty:
    line1 = ax1.plot(episode_stats[ep_col], episode_stats['Avg_Max_Q'], alpha=0.3, color='orange', label='Q-Value từng tập')
    line2 = ax1.plot(episode_stats[ep_col], episode_stats['Moving_Q'], color='red', linewidth=2, label='Q-Value trung bình (20 tập)')
    ax1.set_ylabel('Giá trị Q-Value', color='red', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='red')

ax1.grid(True, linestyle='--', alpha=0.6)

if 'Epsilon' in episode_stats.columns:
    ax2 = ax1.twinx()
    line3 = ax2.plot(episode_stats[ep_col], episode_stats['Epsilon'], color='purple', linewidth=2.5, linestyle='--', label='Epsilon')
    ax2.set_ylabel('Tỷ lệ khám phá (Epsilon)', color='purple', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(-0.05, 1.05)

ax1.set_title('1. Tương quan giữa Tỷ lệ Khám phá (Epsilon) và Q-Value của D3QN (MuJoCo)', pad=15)

# 2. Biểu đồ Phân phối Lựa chọn Hành động
ax_bar = axes[1]
if act_col and act_col in df.columns:
    action_counts = df[act_col].value_counts()
    ax_bar.bar(action_counts.index, action_counts.values, color='teal', alpha=0.85)
    ax_bar.set_title('2. Phân phối lựa chọn Hành động (Action Distribution) của D3QN')
    ax_bar.set_ylabel('Số lần chọn')
    ax_bar.tick_params(axis='x', rotation=30)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout(h_pad=3.0)

out_path = os.path.join(base_dir, "ana3_d3qn_qval_action.png")
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị Q-Value & Action vào: {out_path}")
plt.show()
