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

file_name = find_csv("log_loss_D3QN.csv")
if not os.path.exists(file_name):
    file_name = find_csv("log_loss_parallel_D3QN.csv")

if not os.path.exists(file_name):
    print(f"❌ Không tìm thấy file Loss log CSV cho D3QN!")
    sys.exit(1)

print(f"📊 Đang đọc dữ liệu Loss từ: {file_name}")
df = pd.read_csv(file_name)

ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else df.columns[0])

window_size = 20
df['Moving_Loss'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.plot(df[ep_col], df['Loss'], alpha=0.3, color='orange', label='Loss từng tập (Gốc)')
plt.plot(df[ep_col], df['Moving_Loss'], color='red', linewidth=2, label=f'Loss Trung bình trượt ({window_size} tập)')
plt.title('Biểu đồ Tiến trình Giảm Loss (Smooth L1 Loss) qua các tập (D3QN MuJoCo)')
plt.xlabel('Tập (Episode)')
plt.ylabel('Giá trị Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

max_l = max(25, df['Loss'].max() + 2)
plt.ylim(-0.5, max_l)
plt.tight_layout()

out_path = os.path.join(base_dir, "ana2_d3qn_loss.png")
plt.savefig(out_path, dpi=300)
print(f"🖼️ Đã lưu đồ thị Loss vào: {out_path}")
plt.show()
