import os
import sys
import pickle
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_buffer(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print(f"📂 Đang nạp ReplayBuffer từ: {file_path} ...")
    with open(file_path, "rb") as f:
        buffer_data = pickle.load(f)

    total = len(buffer_data)
    print(f"✅ Đã nạp thành công! Tổng số transition: {total}")
    if total == 0:
        return

    action_names = ["Đi thẳng", "Đi phải", "Đi trái", "Đi chéo trái", "Đi chéo phải"]
    actions = [item[2] for item in buffer_data]
    rewards = [item[3] for item in buffer_data]
    dones = [item[6] for item in buffer_data]

    print("\n📊 === THỐNG KÊ HÀNH ĐỘNG (ACTION DISTRIBUTION) ===")
    for a_idx, name in enumerate(action_names):
        count = actions.count(a_idx)
        pct = (count / total) * 100
        print(f"  • {name} (Action {a_idx}): {count} lần ({pct:.1f}%)")

    print("\n💰 === THỐNG KÊ REWARD ===")
    print(f"  • Min Reward: {np.min(rewards):.2f}")
    print(f"  • Max Reward: {np.max(rewards):.2f}")
    print(f"  • Mean Reward: {np.mean(rewards):.2f}")
    
    collisions = [i for i, r in enumerate(rewards) if r <= -100]
    print(f"  • Số transition bị phạt va chạm (Reward <= -100): {len(collisions)} lần")

    if collisions:
        print("\n💥 === CHI TIẾT CÁC MẪU VA CHẠM ĐIỂN HÌNH ===")
        sample_indices = collisions[:min(3, len(collisions))]
        for rank, idx in enumerate(sample_indices, 1):
            s_img, s_vec, a, r, ns_img, ns_vec, d = buffer_data[idx]
            vec = s_vec.squeeze()
            lidar_m = vec[14:23] * 5.0
            print(f"\n[Mẫu va chạm #{rank} - Bước {idx}]")
            print(f"  - Hành động đã chọn: {action_names[a]} (Action {a})")
            print(f"  - Reward: {r:.2f} | Done: {bool(d)}")
            print(f"  - Rel Goal (dx, dy, dz): {vec[6:9].round(2)}")
            print(f"  - 📡 9 Tia LiDAR (m): [L90:{lidar_m[0]:.2f}m | L67:{lidar_m[1]:.2f}m | L45:{lidar_m[2]:.2f}m | L22:{lidar_m[3]:.2f}m | "
                  f"F0:{lidar_m[4]:.2f}m | R22:{lidar_m[5]:.2f}m | R45:{lidar_m[6]:.2f}m | R67:{lidar_m[7]:.2f}m | R90:{lidar_m[8]:.2f}m]")

    s_img, s_vec, a, r, ns_img, ns_vec, d = buffer_data[-1]
    vec = s_vec.squeeze()
    lidar_m = vec[14:23] * 5.0
    print("\n📍 === MẪU MỚI NHẤT TRONG BUFFER ===")
    print(f"  - Hành động đã chọn: {action_names[a]} (Action {a})")
    print(f"  - Reward: {r:.2f} | Done: {bool(d)}")
    print(f"  - 📡 LiDAR (m): [L90:{lidar_m[0]:.2f}m | L67:{lidar_m[1]:.2f}m | L45:{lidar_m[2]:.2f}m | L22:{lidar_m[3]:.2f}m | "
          f"F0:{lidar_m[4]:.2f}m | R22:{lidar_m[5]:.2f}m | R45:{lidar_m[6]:.2f}m | R67:{lidar_m[7]:.2f}m | R90:{lidar_m[8]:.2f}m]")
    print("===================================================\n")

if __name__ == "__main__":
    default_pkl = os.path.join(current_dir, "Parallel", "Model", "replay_buffer_parallel_eposide100.pkl")
    target = sys.argv[1] if len(sys.argv) > 1 else default_pkl
    analyze_buffer(target)
