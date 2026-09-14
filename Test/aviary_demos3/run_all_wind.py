"""Script tổng hợp chạy kiểm tra toàn bộ các kịch bản của WindWrapper:
1. CONSTANT (Gió thổi liên tục theo vector cố định)
2. GUST (Gió giật ngẫu nhiên theo xung lực đột ngột)
3. DRYDEN (Nhiễu loạn khí động học tiêu chuẩn hàng không MIL-F-8785C)
4. SINUSOIDAL (Gió dao động điều hòa hình sin theo chu kỳ)
5. COMBINED (Gió bão kết hợp: Gió nền + Dryden + Gió giật)
6. COMPARE (Đánh giá đối chuẩn: Không gió vs Có gió)

Cách sử dụng:
    python run_all_wind.py                          # Chạy kiểm tra nhanh toàn bộ
    python run_all_wind.py --record                 # Chạy và xuất ảnh động GIF
    python run_all_wind.py --model constant --gui   # Mở 3D GUI xem riêng kịch bản Constant
    python run_all_wind.py --model gust --gui       # Mở 3D GUI xem riêng kịch bản Gust
    python run_all_wind.py --model dryden --gui     # Mở 3D GUI xem riêng kịch bản Dryden
    python run_all_wind.py --model sinusoidal --gui # Mở 3D GUI xem riêng kịch bản Sinusoidal
    python run_all_wind.py --model combined --gui   # Mở 3D GUI xem riêng kịch bản Combined
    python run_all_wind.py --model compare          # Chạy bài so sánh Benchmark
"""

import argparse
import sys
import os
from importlib import import_module

WIND_MAP = {
    "constant": ("01_test_wind_constant", "run_constant_wind_demo", "Mô hình gió thổi liên tục (CONSTANT)"),
    "gust": ("02_test_wind_gust", "run_gust_wind_demo", "Mô hình gió giật bất ngờ (GUST)"),
    "dryden": ("03_test_wind_dryden", "run_dryden_wind_demo", "Mô hình nhiễu loạn khí quyển (DRYDEN)"),
    "sinusoidal": ("04_test_wind_sinusoidal", "run_sinusoidal_wind_demo", "Mô hình gió điều hòa hình sin (SINUSOIDAL)"),
    "combined": ("05_test_wind_combined", "run_combined_wind_demo", "Mô hình gió bão kết hợp cực hạn (COMBINED)"),
    "visible": ("07_test_wind_drift_visible", "run_visible_wind_demo", "Minh chứng trực quan lực gió & độ trôi dạt (VISIBLE)"),
    "swirls": ("08_test_wind_swirls_icon", "run_swirls_demo", "Trực quan hóa luồng gió xoáy 3D (WIND SWIRLS ICON)"),
}


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm tra wrapper WindWrapper trong MuJoCo-Drones")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "constant", "gust", "dryden", "sinusoidal", "combined", "compare", "visible", "swirls"],
        help="Chọn mô hình gió cần kiểm tra (mặc định: 'all')",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Mở giao diện đồ họa 3D trực quan MuJoCo",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Ghi hình và xuất file GIF vào thư mục gifs/",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Số bước mô phỏng cho mỗi kịch bản (mặc định: 250 bước)",
    )
    args = parser.parse_args()

    if args.model == "compare":
        mod = import_module("06_compare_wind_vs_nowind")
        mod.run_comparison_demo(gui=args.gui, record=args.record, steps=args.steps)
        return

    selected = list(WIND_MAP.keys()) if args.model == "all" else [args.model]

    print("\n" + "#" * 80)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH KIỂM TRA WINDWRAPPER (NHIỄU LOẠN KHÍ ĐỘNG HỌC & GIÓ)")
    print(f" Danh sách kịch bản ({len(selected)}): {[WIND_MAP[k][2] for k in selected]}")
    print(f" Chế độ: GUI={args.gui} | Ghi GIF={args.record} | Số bước={args.steps}")
    print("#" * 80 + "\n")

    for key in selected:
        mod_name, func_name, display_name = WIND_MAP[key]
        print(f"\n>>> [BẮT ĐẦU TEST] {display_name} <<<")
        try:
            mod = import_module(mod_name)
            run_func = getattr(mod, func_name)
            run_func(
                gui=args.gui,
                record=args.record,
                steps=args.steps
            )
        except Exception as e:
            print(f"[LỖI KHI CHẠY {display_name}]: {e}")
            import traceback
            traceback.print_exc()

    if args.model == "all":
        print("\n>>> [CHẠY BỔ SUNG BÀI ĐỐI CHUẨN BENCHMARK] <<<")
        try:
            mod_comp = import_module("06_compare_wind_vs_nowind")
            mod_comp.run_comparison_demo(gui=args.gui, record=args.record, steps=args.steps)
        except Exception as e:
            print(f"[LỖI KHI CHẠY BENCHMARK]: {e}")

    print("\n" + "#" * 80)
    print(" HOÀN THÀNH TẤT CẢ CÁC BÀI TEST WINDWRAPPER!")
    if args.record:
        gif_dir = os.path.join(os.path.dirname(__file__), "gifs")
        print(f" Toàn bộ các file GIF đã được xuất vào: {gif_dir}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
