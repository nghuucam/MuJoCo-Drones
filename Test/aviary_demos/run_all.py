"""Script tổng hợp để chạy kiểm tra tất cả 5 môi trường:
1. VelocityAviary (Bám đuổi vận tốc)
2. FlyThroughAviary (Bay qua các điểm waypoint)
3. FormationAviary (Bay đội hình)
4. RaceAviary (Đua qua cổng)
5. MultiAgentAviary (Chuẩn PettingZoo đa tác tử)

Cách sử dụng:
    python run_all.py --help
    python run_all.py                        # Chạy kiểm tra nhanh lần lượt cả 5 môi trường
    python run_all.py --record               # Chạy và xuất file ảnh động GIF cho cả 5 môi trường
    python run_all.py --gui --env velocity   # Mở GUI xem riêng môi trường VelocityAviary
    python run_all.py --gui --env fly_through# Mở GUI xem riêng môi trường FlyThroughAviary
    python run_all.py --gui --env formation  # Mở GUI xem riêng môi trường FormationAviary
    python run_all.py --gui --env race       # Mở GUI xem riêng môi trường RaceAviary
    python run_all.py --gui --env multi_agent# Mở GUI xem riêng môi trường MultiAgentAviary
"""

import argparse
import sys
import os

# Import các hàm chạy demo
from importlib import import_module

ENV_MAP = {
    "velocity": ("01_test_velocity_aviary", "run_velocity_demo", "VelocityAviary (Bám đuổi vận tốc)"),
    "fly_through": ("02_test_fly_through_aviary", "run_fly_through_demo", "FlyThroughAviary (Bay qua Waypoints)"),
    "formation": ("03_test_formation_aviary", "run_formation_demo", "FormationAviary (Bay đội hình)"),
    "race": ("04_test_race_aviary", "run_race_demo", "RaceAviary (Đua qua cổng)"),
    "multi_agent": ("05_test_multi_agent_aviary", "run_multi_agent_demo", "MultiAgentAviary (PettingZoo Đa tác tử)"),
}


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm tra các môi trường MuJoCo Drones Aviary")
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=["all", "velocity", "fly_through", "formation", "race", "multi_agent"],
        help="Chọn môi trường cần chạy (mặc định: 'all' - chạy lần lượt cả 5)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Bật giao diện đồ họa 3D trực quan MuJoCo (render_mode='human')",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Ghi hình và lưu file hoạt ảnh .gif vào thư mục Test/aviary_demos/gifs/",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Số bước mô phỏng cho mỗi môi trường (mặc định: 250 bước)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sử dụng hành động ngẫu nhiên (action sample) thay vì điều khiển PID",
    )
    args = parser.parse_args()

    selected_envs = list(ENV_MAP.keys()) if args.env == "all" else [args.env]

    print("\n" + "#" * 80)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH KIỂM TRA MÔI TRƯỜNG MUJOCO-DRONES AVIARY")
    print(f" Danh sách môi trường sẽ chạy ({len(selected_envs)}): {[ENV_MAP[k][2] for k in selected_envs]}")
    print(f" Chế độ: GUI={args.gui} | Ghi GIF={args.record} | Số bước={args.steps} | Hành động={'Ngẫu nhiên' if args.random else 'PID Tracking'}")
    print("#" * 80 + "\n")

    for key in selected_envs:
        mod_name, func_name, display_name = ENV_MAP[key]
        print(f"\n>>> [BẮT ĐẦU TEST] {display_name} <<<")
        try:
            mod = import_module(mod_name)
            run_func = getattr(mod, func_name)
            run_func(
                gui=args.gui,
                record=args.record,
                steps=args.steps,
                use_random=args.random
            )
        except Exception as e:
            print(f"[LỖI KHI CHẠY {display_name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#" * 80)
    print(" HOÀN THÀNH TẤT CẢ CÁC BÀI TEST MÔI TRƯỜNG!")
    if args.record:
        gif_dir = os.path.join(os.path.dirname(__file__), "gifs")
        print(f" Các file GIF đã được tạo trong: {gif_dir}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
