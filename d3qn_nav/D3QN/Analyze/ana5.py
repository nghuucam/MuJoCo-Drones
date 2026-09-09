import os
import sys
import pandas as pd

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

def analyze_flight_log(csv_file):
    if not os.path.exists(csv_file):
        print(f"❌ Lỗi: Không tìm thấy file '{csv_file}'.")
        return
        
    print(f"📊 BÁO CÁO CHI TIẾT TỪ FILE: {csv_file}")
    df = pd.read_csv(csv_file)
    
    ep_col = 'Episode' if 'Episode' in df.columns else ('Eposide' if 'Eposide' in df.columns else 'Step')
    if ep_col in df.columns and df[ep_col].duplicated().any():
        episodes = df.groupby(ep_col).last().reset_index()
    else:
        episodes = df.copy()

    total_episodes = len(episodes)
    wins = (episodes['Win'].astype(str) == '1').sum()
    losses = total_episodes - wins
    
    collisions = (episodes['Collision'].astype(str) == '1').sum() if 'Collision' in episodes.columns else 0
    over_maps = (episodes['Over_Map'].astype(str) == '1').sum() if 'Over_Map' in episodes.columns else 0
    over_steps = (episodes['Over_Step'].astype(str) == '1').sum() if 'Over_Step' in episodes.columns else 0
    
    print("=" * 60)
    print(f"📋 BÁO CÁO THỐNG KÊ TỔNG THỂ MODEL D3QN (MUJOCO)")
    print("=" * 60)
    print(f"Tổng số tập đã thử nghiệm (Episodes) : {total_episodes}")
    print(f"🏆 Số lần THẮNG (Tới đích thành công) : {wins} ({(wins/max(1, total_episodes))*100:.2f}%)")
    print(f"💥 Số lần THUA                       : {losses} ({(losses/max(1, total_episodes))*100:.2f}%)")
    print("-" * 60)
    print("Chi tiết các nguyên nhân kết thúc màn chơi:")
    print(f"  - Thua do Va chạm vật cản   : {collisions} lần ({(collisions/max(1, total_episodes))*100:.2f}%)")
    print(f"  - Thua do Bay ra ngoài map  : {over_maps} lần ({(over_maps/max(1, total_episodes))*100:.2f}%)")
    print(f"  - Thua do Vượt quá số bước  : {over_steps} lần ({(over_steps/max(1, total_episodes))*100:.2f}%)")
    
    if 'AccumReward' in episodes.columns:
        print("-" * 60)
        print(f"Điểm thưởng (Reward) Trung bình : {episodes['AccumReward'].mean():.2f}")
        print(f"Điểm thưởng Cao nhất            : {episodes['AccumReward'].max():.2f}")
        print(f"Điểm thưởng Thấp nhất           : {episodes['AccumReward'].min():.2f}")
    print("=" * 60)

if __name__ == "__main__":
    file_name = find_csv("drone_flight_log_test_D3QN.csv")
    if not os.path.exists(file_name):
        file_name = find_csv("drone_flight_log_parallel_D3QN.csv")

    analyze_flight_log(file_name)
