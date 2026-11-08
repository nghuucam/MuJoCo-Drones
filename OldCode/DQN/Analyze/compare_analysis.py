import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess(csv_path):
    """Đọc và chuẩn hóa dữ liệu từ file CSV flight log."""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    episode_col = 'Episode' if 'Episode' in df.columns else 'Eposide'
    if episode_col not in df.columns:
        return None
        
    df[episode_col] = pd.to_numeric(df[episode_col], errors='coerce')
    episodes = df.groupby(episode_col).last().reset_index()
    
    for col in ['Win', 'Collision', 'Over_Step', 'Over_Map']:
        if col in episodes.columns:
            episodes[col] = episodes[col].astype(str).str.strip().isin(['1', 'True', '1.0'])
        else:
            episodes[col] = False
            
    if 'AccumReward' in episodes.columns:
        episodes['AccumReward'] = pd.to_numeric(episodes['AccumReward'], errors='coerce')
    else:
        episodes['AccumReward'] = 0.0

    return episodes, episode_col

def compare_two_logs(file1, name1, file2, name2, window_size=30):
    """So sánh và vẽ biểu đồ phân tích giữa 2 file flight log CSV cho DQN."""
    res1 = load_and_preprocess(file1)
    res2 = load_and_preprocess(file2)
    
    if res1 is None and res2 is None:
        print(f"❌ Không tìm thấy hoặc lỗi cấu trúc cả 2 file: '{file1}' và '{file2}'")
        return
    
    print("=" * 65)
    print(f"📊 BẢNG SO SÁNH HIỆU NĂNG HUẤN LUYỆN STANDARD DQN")
    print("=" * 65)
    print(f"{'Chỉ số đánh giá':<30} | {name1:<15} | {name2:<15}")
    print("-" * 65)

    def print_metrics(episodes, name):
        if episodes is None or len(episodes) == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0
        tot = len(episodes)
        win_rate = (episodes['Win'].sum() / tot) * 100
        coll_rate = (episodes['Collision'].sum() / tot) * 100
        over_step = (episodes['Over_Step'].sum() / tot) * 100
        over_map = (episodes['Over_Map'].sum() / tot) * 100
        avg_rew = episodes['AccumReward'].mean()
        return tot, win_rate, coll_rate, over_step, over_map, avg_rew

    t1, w1, c1, os1, om1, r1 = print_metrics(res1[0] if res1 else None, name1)
    t2, w2, c2, os2, om2, r2 = print_metrics(res2[0] if res2 else None, name2)

    print(f"{'Tổng số Episode':<30} | {t1:<15} | {t2:<15}")
    print(f"{'Tỷ lệ Thắng (Win Rate)':<30} | {w1:>14.2f}% | {w2:>14.2f}%")
    print(f"{'Tỷ lệ Va chạm (Collision)':<30} | {c1:>14.2f}% | {c2:>14.2f}%")
    print(f"{'Tỷ lệ Quá bước (Over Step)':<30} | {os1:>14.2f}% | {os2:>14.2f}%")
    print(f"{'Tỷ lệ Ra khỏi Map (Over Map)':<30} | {om1:>14.2f}% | {om2:>14.2f}%")
    print(f"{'Điểm trung bình (Avg Reward)':<30} | {r1:>15.2f} | {r2:>15.2f}")
    print("=" * 65)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'So sánh Kết quả Huấn luyện DQN: {name1} vs {name2}', fontsize=16, fontweight='bold')

    # 1. Biểu đồ Điểm thưởng (Accumulated Reward)
    ax1 = axes[0, 0]
    if res1 is not None:
        ep1, col1 = res1
        ma_r1 = ep1['AccumReward'].rolling(window=window_size, min_periods=1).mean()
        ax1.plot(ep1[col1], ma_r1, label=f'{name1} (Moving Avg {window_size})', color='blue', linewidth=2)
        ax1.plot(ep1[col1], ep1['AccumReward'], alpha=0.15, color='blue')
    
    if res2 is not None:
        ep2, col2 = res2
        ma_r2 = ep2['AccumReward'].rolling(window=window_size, min_periods=1).mean()
        ax1.plot(ep2[col2], ma_r2, label=f'{name2} (Moving Avg {window_size})', color='orange', linewidth=2)
        ax1.plot(ep2[col2], ep2['AccumReward'], alpha=0.15, color='orange')
        
    ax1.set_title('1. Tiến trình Điểm thưởng (Accumulated Reward)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. Biểu đồ Tỷ lệ thắng trượt
    ax2 = axes[0, 1]
    if res1 is not None:
        ep1, col1 = res1
        ma_w1 = ep1['Win'].rolling(window=window_size, min_periods=1).mean() * 100
        ax2.plot(ep1[col1], ma_w1, label=f'{name1}', color='green', linewidth=2)
        
    if res2 is not None:
        ep2, col2 = res2
        ma_w2 = ep2['Win'].rolling(window=window_size, min_periods=1).mean() * 100
        ax2.plot(ep2[col2], ma_w2, label=f'{name2}', color='red', linewidth=2)

    ax2.set_title(f'2. Tỷ lệ Chiến thắng trượt (Moving Win Rate - {window_size} eps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 3. Biểu đồ Cột phân bố kết quả
    ax3 = axes[1, 0]
    categories = ['Thắng (Win)', 'Va chạm', 'Hết bước', 'Ra ngoài map']
    vals1 = [w1, c1, os1, om1]
    vals2 = [w2, c2, os2, om2]

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(x - width/2, vals1, width, label=name1, color='skyblue')
    ax3.bar(x + width/2, vals2, width, label=name2, color='sandybrown')

    ax3.set_title('3. Phân bố Kết quả Kết thúc Episode (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Tỷ lệ (%)')
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.legend()

    # 4. Biểu đồ So sánh Loss
    ax4 = axes[1, 1]
    loss_file1 = file1.replace("drone_flight_log", "log_loss")
    loss_file2 = file2.replace("drone_flight_log", "log_loss")

    has_loss = False
    if os.path.exists(loss_file1):
        df_l1 = pd.read_csv(loss_file1)
        ep_c = 'Episode' if 'Episode' in df_l1.columns else 'Eposide'
        if ep_c in df_l1.columns and 'Loss' in df_l1.columns:
            l1_ma = df_l1['Loss'].rolling(window=window_size, min_periods=1).mean()
            ax4.plot(df_l1[ep_c], l1_ma, label=f'Loss {name1}', color='purple', linewidth=2)
            has_loss = True

    if os.path.exists(loss_file2):
        df_l2 = pd.read_csv(loss_file2)
        ep_c = 'Episode' if 'Episode' in df_l2.columns else 'Eposide'
        if ep_c in df_l2.columns and 'Loss' in df_l2.columns:
            l2_ma = df_l2['Loss'].rolling(window=window_size, min_periods=1).mean()
            ax4.plot(df_l2[ep_c], l2_ma, label=f'Loss {name2}', color='crimson', linewidth=2)
            has_loss = True

    if has_loss:
        ax4.set_title('4. Tiến trình Giảm Loss (Smooth L1 Loss)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Chưa có dữ liệu Log Loss', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax4.set_title('4. Tiến trình Giảm Loss')

    chart_path = os.path.join(base_dir, "dqn_comparison_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    print(f"\n🖼️  Đã lưu biểu đồ so sánh DQN vào: {chart_path}")
    
    try:
        plt.show()
    except Exception:
        pass

def find_csv_file(base_dir, parent_dir, filename, sub_folder):
    candidates = [
        os.path.join(parent_dir, sub_folder, filename),
        os.path.join(parent_dir, filename),
        os.path.join(base_dir, filename),
        os.path.join(base_dir, sub_folder, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(parent_dir, sub_folder, filename)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    FILE_1 = find_csv_file(base_dir, parent_dir, "drone_flight_log_test_DQN.csv", "Single")
    NAME_1 = "DQN Đơn lẻ (Single)"

    FILE_2 = find_csv_file(base_dir, parent_dir, "drone_flight_log_parallel_DQN.csv", "Parallel")
    NAME_2 = "DQN Song song (Parallel)"

    print(f"Đang tiến hành phân tích và so sánh 2 file DQN:")
    print(f"1. {FILE_1}")
    print(f"2. {FILE_2}\n")

    compare_two_logs(FILE_1, NAME_1, FILE_2, NAME_2, window_size=30)
