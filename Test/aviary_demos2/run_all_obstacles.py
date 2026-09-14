"""Script tổng hợp để chạy kiểm tra toàn bộ 5 chủ đề môi trường Chướng ngại vật:
1. FOREST (Rừng cây: Thân cây gỗ + Tán lá xanh)
2. URBAN (Đô thị: Các khối tòa nhà cao tầng)
3. INDOOR (Trong nhà: Phòng kín, 4 vách tường, trần nhà, bàn ghế, cột trụ)
4. RANDOM (Ngẫu nhiên: Hỗn hợp khối cầu, trụ, hộp xoay góc tự do)
5. GATES (Cổng đua: Hệ thống cổng đua liên hoàn theo mạch vòng)

Cách sử dụng:
    python run_all_obstacles.py                           # Chạy kiểm tra nhanh 5 chủ đề
    python run_all_obstacles.py --record                  # Chạy và xuất hoạt ảnh GIF cho cả 5 chủ đề
    python run_all_obstacles.py --theme forest --gui      # Mở 3D GUI xem riêng chủ đề FOREST
    python run_all_obstacles.py --theme urban --gui       # Mở 3D GUI xem riêng chủ đề URBAN
    python run_all_obstacles.py --theme indoor --gui      # Mở 3D GUI xem riêng chủ đề INDOOR
    python run_all_obstacles.py --theme random --gui      # Mở 3D GUI xem riêng chủ đề RANDOM
    python run_all_obstacles.py --theme gates --gui       # Mở 3D GUI xem riêng chủ đề GATES
"""

import argparse
import sys
import os
from importlib import import_module

THEME_MAP = {
    "forest": ("01_test_obstacle_forest", "run_forest_demo", "Chủ đề Rừng cây (FOREST)"),
    "urban": ("02_test_obstacle_urban", "run_urban_demo", "Chủ đề Đô thị (URBAN)"),
    "indoor": ("03_test_obstacle_indoor", "run_indoor_demo", "Chủ đề Trong nhà (INDOOR)"),
    "random": ("04_test_obstacle_random", "run_random_demo", "Chủ đề Ngẫu nhiên hỗn hợp (RANDOM)"),
    "gates": ("05_test_obstacle_gates", "run_gates_demo", "Chủ đề Cổng đua liên hoàn (GATES)"),
}


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm tra các môi trường Chướng ngại vật MuJoCo-Drones")
    parser.add_argument(
        "--theme",
        type=str,
        default="all",
        choices=["all", "forest", "urban", "indoor", "random", "gates"],
        help="Chọn chủ đề cần chạy (mặc định: 'all' - chạy lần lượt cả 5 chủ đề)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Mở giao diện đồ họa 3D trực quan tương tác của MuJoCo",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Ghi hình và xuất tệp ảnh động .gif vào thư mục Test/aviary_demos2/gifs/",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Số bước mô phỏng cho mỗi chủ đề (mặc định: 250 bước)",
    )
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=20,
        help="Số lượng chướng ngại vật sinh ngẫu nhiên",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên",
    )
    args = parser.parse_args()

    selected_themes = list(THEME_MAP.keys()) if args.theme == "all" else [args.theme]

    print("\n" + "#" * 80)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH KIỂM TRA MÔI TRƯỜNG CHƯỚNG NGẠI VẬT (OBSTACLES)")
    print(f" Danh sách chủ đề ({len(selected_themes)}): {[THEME_MAP[k][2] for k in selected_themes]}")
    print(f" Chế độ: GUI={args.gui} | Ghi GIF={args.record} | Số bước={args.steps} | Số vật cản={args.num_obstacles}")
    print("#" * 80 + "\n")

    for key in selected_themes:
        mod_name, func_name, display_name = THEME_MAP[key]
        print(f"\n>>> [BẮT ĐẦU TEST] {display_name} <<<")
        try:
            mod = import_module(mod_name)
            run_func = getattr(mod, func_name)
            run_func(
                gui=args.gui,
                record=args.record,
                steps=args.steps,
                num_obstacles=args.num_obstacles,
                seed=args.seed
            )
        except Exception as e:
            print(f"[LỖI KHI CHẠY {display_name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#" * 80)
    print(" HOÀN THÀNH TẤT CẢ CÁC BÀI TEST CHƯỚNG NGẠI VẬT!")
    if args.record:
        gif_dir = os.path.join(os.path.dirname(__file__), "gifs")
        print(f" Toàn bộ các file GIF đã được xuất vào: {gif_dir}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
