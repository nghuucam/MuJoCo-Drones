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
    os.path.join(oldcode_dir, "D3QN", "Parallel", "log_loss_parallel_D3QN.csv"),
    os.path.join(oldcode_dir, "D3QN", "Single", "log_loss_D3QN.csv"),
    os.path.join(parent_dir, "Parallel", "log_loss_parallel_D3QN.csv"),
    os.path.join(parent_dir, "Single", "log_loss_D3QN.csv")
]

dqn_candidates = [
    os.path.join(oldcode_dir, "DQN", "Parallel", "log_loss_parallel_DQN.csv"),
    os.path.join(oldcode_dir, "DQN", "Single", "log_loss_DQN.csv")
]

f_loss_d3qn = find_file(d3qn_candidates)
f_loss_dqn = find_file(dqn_candidates)

if not f_loss_d3qn and not f_loss_dqn:
    print("❌ Không tìm thấy file Loss log CSV nào!")
    sys.exit(1)

df_d3qn = pd.read_csv(f_loss_d3qn) if f_loss_d3qn else None
df_dqn = pd.read_csv(f_loss_dqn) if f_loss_dqn else None

print("=" * 70)
print("📊 BẢNG SO SÁNH TIẾN TRÌNH GIẢM LOSS (ANA2: TRAINING LOSS)")
print("=" * 70)
print(f"{'Thuật toán':<20} | {'Loss Trung bình':<16} | {'Loss Min':<12} | {'Loss Max':<12}")
print("-" * 70)

if df_d3qn is not None:
    print(f"{'D3QN':<20} | {df_d3qn['Loss'].mean():<16.4f} | {df_d3qn['Loss'].min():<12.4f} | {df_d3qn['Loss'].max():<12.4f}")
if df_dqn is not None:
    print(f"{'DQN':<20} | {df_dqn['Loss'].mean():<16.4f} | {df_dqn['Loss'].min():<12.4f} | {df_dqn['Loss'].max():<12.4f}")
print("=" * 70)

window_size = 20

plt.figure(figsize=(11, 6))

if df_d3qn is not None:
    ep_col_d3qn = df_d3qn.columns[0]
    df_d3qn['MA_Loss'] = df_d3qn['Loss'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df_d3qn[ep_col_d3qn], df_d3qn['Loss'], alpha=0.15, color='#2ecc71')
    plt.plot(df_d3qn[ep_col_d3qn], df_d3qn['MA_Loss'], color='#27ae60', linewidth=2.5, label=f'D3QN Loss (SMA {window_size})')

if df_dqn is not None:
    ep_col_dqn = df_dqn.columns[0]
    df_dqn['MA_Loss'] = df_dqn['Loss'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df_dqn[ep_col_dqn], df_dqn['Loss'], alpha=0.15, color='#e74c3c')
    plt.plot(df_dqn[ep_col_dqn], df_dqn['MA_Loss'], color='#c0392b', linewidth=2.5, linestyle='--', label=f'DQN Loss (SMA {window_size})')

plt.title('So sánh Tiến trình Giảm Loss (Smooth L1 Loss) giữa D3QN và DQN', fontsize=13, fontweight='bold')
plt.xlabel('Tập (Episode)', fontsize=11)
plt.ylabel('Giá trị Loss', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)

out_path = os.path.join(base_dir, "ana2_compare_loss.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị Loss vào: {out_path}")
