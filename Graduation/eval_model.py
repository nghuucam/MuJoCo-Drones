import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO

MODELS_DIR = os.path.join(current_dir, "models")


def get_default_model_path():
    final_path = os.path.join(MODELS_DIR, "drone_ppo_curriculum_ver4_final.zip")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(MODELS_DIR):
        zips = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
        if zips:
            zips.sort(key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True)
            return os.path.join(MODELS_DIR, zips[0])
    return final_path


def evaluate():
    parser = argparse.ArgumentParser(description="Đánh giá / Xem thử Drone hoạt động với Model đã huấn luyện (Ver4)")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file model (.zip). Mặc định: tự tìm model mới nhất")
    parser.add_argument("--level",      type=int, default=6,    help="Cấp độ Curriculum cần test (0 đến 6, Mặc định: 6)")
    parser.add_argument("--episodes",   type=int, default=5,    help="Số lượng episode test (Mặc định: 5)")
    parser.add_argument("--no-gui",     action="store_true",    help="Tắt giao diện 3D (Chỉ in kết quả log)")
    parser.add_argument("--show-marker",action="store_true",    help="Bật lại con trỏ chấm trắng khổng lồ trên đầu Drone")
    parser.add_argument("--quiet",      action="store_true",    help="Ẩn bảng in tọa độ chi tiết từng bước")
    args = parser.parse_args()

    model_path = args.model_path if (args.model_path and os.path.exists(args.model_path)) else get_default_model_path()

    print("=" * 80)
    print("🛸 ĐÁNH GIÁ MÔ HÌNH DRONE HUẤN LUYỆN VER4 (EVALUATION)")
    print("=" * 80)
    print(f"📦 Đang nạp Model từ: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY FILE MODEL TẠI: {model_path}")
        return

    gui = not args.no_gui
    print(f"🎮 Giao diện 3D GUI : {'BẬT' if gui else 'TẮT'}")
    print(f"🎯 Cấp độ thử nghiệm: Level {args.level}")
    print("-" * 80)

    env = DronePPOCurriculumEnv(gui=gui, show_marker=args.show_marker)
    env.set_level(args.level)

    model = PPO.load(model_path, env=env)
    print("✅ Đã nạp thành công mô hình PPO!")

    success_count = 0
    # Mỗi phần tử: (ep_reward, result_type, final_dist, step_count)
    results = []

    for ep in range(args.episodes):
        print(f"\n▶️ --- EPISODE TEST {ep + 1}/{args.episodes} (Level {args.level}) ---")
        obs, info = env.reset()

        goal_p    = env.goal[0].copy()
        cur_p     = env.pos[0].copy()
        init_dist = float(np.linalg.norm(cur_p - goal_p))
        print(f"   🎯 Tọa độ Đích      : X={goal_p[0]:+6.2f}, Y={goal_p[1]:+6.2f}, Z={goal_p[2]:5.2f}m")
        print(f"   🛫 Vị trí Xuất phát : X={cur_p[0]:+6.2f}, Y={cur_p[1]:+6.2f}, Z={cur_p[2]:5.2f}m  (Cách đích: {init_dist:.2f}m)")

        if not args.quiet:
            print(f"   {'Bước':<6} | {'Tọa độ Drone [X, Y, Z]':<26} | {'Cách đích':<10} | {'Độ cao ΔZ':<10} | {'Reward':<8}")
            print(f"   {'-'*6} + {'-'*26} + {'-'*10} + {'-'*10} + {'-'*8}")

        ep_reward  = 0.0
        step_count = 0

        for step in range(config.MAX_STEPS):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            cur_p     = env.pos[0]
            dist_goal = float(np.linalg.norm(cur_p - goal_p))
            delta_z   = float(abs(cur_p[2] - goal_p[2]))

            ep_reward  += reward
            step_count += 1

            if not args.quiet:
                coord_str = f"[{cur_p[0]:+6.2f}, {cur_p[1]:+6.2f}, {cur_p[2]:5.2f}m]"
                print(f"   #{step_count:03d}   | {coord_str:<26} | {dist_goal:7.2f}m   | {delta_z:7.2f}m   | {reward:+7.2f}")

            if hasattr(env, "_fpv_window") and env._fpv_window is not None:
                try:
                    env._fpv_window.title(
                        f"FPV | Drone: [{cur_p[0]:+.2f}, {cur_p[1]:+.2f}, {cur_p[2]:.2f}] | Đích: {dist_goal:.2f}m"
                    )
                except Exception:
                    pass

            if gui:
                time.sleep(0.08)

            if terminated or truncated:
                break

        is_win      = bool(info.get("win",       False))
        is_col      = bool(info.get("collision",  False))
        is_overmap  = bool(info.get("over_map",   False))
        is_overstep = bool(info.get("over_step",  False))

        cur_p      = env.pos[0]
        final_dist = float(np.linalg.norm(cur_p - goal_p))

        if is_win:
            success_count += 1
            result_type = "win"
            status_str  = "🎉 THÀNH CÔNG (Tới Đích!)"
        elif is_col:
            result_type = "collision"
            status_str  = "💥 VA CHẠM (Đâm Cột / Gai / Sàn)"
        elif is_overmap:
            result_type = "over_map"
            status_str  = "🚩 VĂNG MAP (Bay ra ngoài biên bản đồ)"
        elif is_overstep or truncated:
            result_type = "over_step"
            status_str  = f"⏱️ HẾT BƯỚC (Đạt tối đa {config.MAX_STEPS} bước - Truncated)"
        else:
            result_type = "other"
            status_str  = "🛑 KẾT THÚC (Không xác định)"

        results.append((ep_reward, result_type, final_dist, step_count))

        print(f"   - Kết quả           : {status_str}")
        print(f"   - Vị trí Drone cuối : X={cur_p[0]:.2f}, Y={cur_p[1]:.2f}, Z={cur_p[2]:.2f}")
        print(f"   - Vị trí Đích       : X={goal_p[0]:.2f}, Y={goal_p[1]:.2f}, Z={goal_p[2]:.2f}")
        print(f"   - Khoảng cách đích  : {final_dist:.2f}m  (Yêu cầu thắng: < {config.GOAL_THRESHOLD}m)")
        print(f"   - Tổng số bước      : {step_count} steps | Phần thưởng: {ep_reward:.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # TỔNG HỢP KẾT QUẢ
    # ──────────────────────────────────────────────────────────────────────────
    n         = args.episodes
    wins      = [(r, d, s) for r, t, d, s in results if t == "win"]
    cols      = [(r, d, s) for r, t, d, s in results if t == "collision"]
    overmaps  = [(r, d, s) for r, t, d, s in results if t == "over_map"]
    oversteps = [(r, d, s) for r, t, d, s in results if t == "over_step"]
    others    = [(r, d, s) for r, t, d, s in results if t == "other"]
    all_rews  = [r for r, _, _, _ in results]

    def _row(label, data):
        cnt   = len(data)
        pct   = cnt / n * 100
        avg_r = float(np.mean([x[0] for x in data])) if data else 0.0
        avg_d = float(np.mean([x[1] for x in data])) if data else 0.0
        avg_s = float(np.mean([x[2] for x in data])) if data else 0.0
        return (
            f"   {label:<36} {cnt:>5}  {pct:>6.1f}%  "
            f"{avg_r:>+10.2f}  {avg_d:>11.2f}m  {avg_s:>8.1f}"
        )

    print("\n" + "=" * 80)
    print("📊 TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 80)
    print(f"   • Tổng số Episode test   : {n}")
    print(f"   • Phần thưởng trung bình : {np.mean(all_rews):.2f}")
    print()
    print(
        f"   {'Loại kết quả':<36} {'SL':>5}  {'Tỉ lệ':>7}  "
        f"{'TB Reward':>10}  {'TB Cách đích':>13}  {'TB Bước':>8}"
    )
    print(f"   {'-'*36} {'-'*5}  {'-'*7}  {'-'*10}  {'-'*13}  {'-'*8}")
    print(_row("🎉 Thành công (tới đích)", wins))
    print(_row("💥 Va chạm (đâm cột/gai/sàn)", cols))
    print(_row("🚩 Văng map (bay ra ngoài biên)", overmaps))
    print(_row("⏱️  Hết bước (timeout)", oversteps))
    if others:
        print(_row("🛑 Khác (không xác định)", others))
    print("=" * 80)

    # Gợi ý phân tích nguyên nhân thất bại
    if cols:
        avg_col_dist = float(np.mean([d for _, d, _ in cols]))
        print(f"\n   💡 Va chạm : Trung bình xảy ra khi còn cách đích {avg_col_dist:.2f}m.")
        print(f"       Gợi ý  : Regress penalty đang hơi mạnh (x3.0), cân nhắc giảm xuống x1.5")
        print(f"                trong hàm _computeSubstepReward() để Drone dám lách gai linh hoạt hơn.")

    if oversteps:
        avg_os_dist = float(np.mean([d for _, d, _ in oversteps]))
        avg_os_step = float(np.mean([s for _, _, s in oversteps]))
        print(f"\n   💡 Hết bước: Trung bình còn cách đích {avg_os_dist:.2f}m sau {avg_os_step:.0f} bước.")
        if avg_os_dist < 5.0:
            print(f"       Drone đã đi được phần lớn quãng đường nhưng chưa kịp về đích.")
            print(f"       Gợi ý  : Cân nhắc tăng MAX_STEPS trong config.py (hiện tại = {config.MAX_STEPS}).")
        else:
            print(f"       Drone bị kẹt hoặc do dự ở giữa đường - có thể do reward lùi phạt quá nặng.")

    if overmaps:
        print(f"\n   💡 Văng map: Drone mất định hướng và bay vượt ra ngoài biên bản đồ.")
        print(f"       Gợi ý  : Kiểm tra lại biên giới hạn Overmap trong config.py.")

    print("=" * 80)

    env.close()


if __name__ == "__main__":
    evaluate()
