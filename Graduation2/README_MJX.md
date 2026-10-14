# HƯỚNG DẪN MÔ PHỎNG VÀ HUẤN LUYỆN DRONE SONG SONG BẰNG MUJOCO MJX (GRADUATION2)

Thư mục **`Graduation2/`** là bản chạy GPU của `Graduation/`, dùng **MuJoCo MJX + JAX** để mô phỏng hàng trăm drone song song và huấn luyện PPO (SBX) với Curriculum 7 cấp độ. Không ảnh hưởng tới mã nguồn trong `Graduation/`.

---

## 📁 Cấu trúc file

| File | Vai trò |
|---|---|
| `config_mjx.py` | Thông số mô phỏng, action, bộ điều khiển, 7 level curriculum, toạ độ 15 cọc gai |
| `model_xml_builder.py` | Sinh XML: drone dạng chất điểm 27 g (3 motor lực theo trục thân), 15 cọc gai mocap, quả cầu đích |
| `drone_mjx_env.py` | Môi trường JAX (`jit` + `vmap`): action → waypoint, bám quỹ đạo mượt, phát hiện va chạm, reward, auto-reset |
| `apf_jax.py` | Gợi ý hướng bay APF (2 chiều cuối của observation) |
| `env_numpy.py` | Bản NumPy tương đương của observation / waypoint / bộ điều khiển, dùng cho viewer trên CPU |
| `mjx_gym_wrapper.py` | Bọc thành `VecEnv` cho SB3 / SBX |
| `curriculum_callback.py` | Tự thăng cấp (≥ 80%) / hạ cấp (≤ 20%) theo 200 episode gần nhất |
| `train_sbx_ppo.py` | **Script huấn luyện chính** |
| `train_mjx_ppo.py` | Benchmark tốc độ mô phỏng bằng action ngẫu nhiên (không huấn luyện) |
| `eval_mjx.py` | Xem 3D trên máy cá nhân (CPU), dùng model `.zip` hoặc phi công APF |
| `warp_renderer.py` | Khung dự phòng cho render ảnh batch bằng Warp (chưa dùng) |

---

## ⚙️ Cơ chế một bước

1. Action `[a, b, d] ∈ [-1, 1]³` → hướng lệch so với heading hiện tại (±35°), góc nâng (±30°), độ dài bước 0.2–1.5 m.
2. Drone quay heading theo hướng di chuyển (giống `Graduation/`), rồi bám quỹ đạo min-jerk tới waypoint trong `SUBSTEPS / SIM_FREQ = 0.5 s`.
3. Thắng / va chạm được kiểm tra ở **từng substep**; va chạm xác định bằng `contact.dist < 0`.

---

## 🚀 Chạy

### Huấn luyện trên Colab (GPU)
```bash
pip install sbx-rl flax "jax[cuda12]" mujoco mujoco-mjx
python Graduation2/train_sbx_ppo.py --exp-name colab_curriculum_sbx --num-envs 128 --timesteps 5000000
```

### Xem 3D trên máy cá nhân
```bash
python Graduation2/eval_mjx.py --level 3
python Graduation2/eval_mjx.py --level 6 --model-path path/to/model.zip
```

### Benchmark tốc độ
```bash
python Graduation2/train_mjx_ppo.py --num-envs 1024 --level 6 --timesteps 1000000
```
