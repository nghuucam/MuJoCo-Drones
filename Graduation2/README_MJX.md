# HƯỚNG DẪN MÔ PHỎNG VÀ HUẤN LUYỆN DRONE SONG SONG BẰNG MUJOCO MJX (GRADUATION2)

Thư mục **`Graduation2/`** là module nghiên cứu độc lập nhằm khai thác sức mạnh của **MuJoCo MJX (Google DeepMind)** và **JAX**:
* Chạy song song **1.024 đến 4.096 drone cùng lúc trực tiếp trên GPU**.
* Tốc độ mô phỏng đạt hàng chục nghìn đến hàng trăm nghìn **FPS**.
* Hoàn toàn độc lập, không làm ảnh hưởng đến mã nguồn chính trong `Graduation/`.

---

## 📁 Cấu Trúc Các File Trong `Graduation2/`

1. **`config_mjx.py`**: Định nghĩa thông số vật lý của Crazyflie 2.X, dải đích 7 cấp độ Curriculum và tọa độ tĩnh của 15 cọc gai.
2. **`model_xml_builder.py`**: Tự động sinh mô hình MuJoCo XML tĩnh có 15 cọc gai và các cành gai nhọn để nạp vào MJX.
3. **`drone_mjx_env.py`**: Môi trường vector hóa 100% bằng JAX (`jit` + `vmap`), tự động tính toán va chạm và hàm phần thưởng chuẩn trên GPU.
4. **`mjx_gym_wrapper.py`**: Wrapper tương thích Gymnasium để cắm vào Stable-Baselines3 hoặc SBX khi cần.
5. **`train_mjx_ppo.py`**: Script thực thi huấn luyện song song tốc độ cao.

---

## 🚀 Hướng Dẫn Chạy Thử Nghiệm

### Bước 1: Cài đặt thư viện cần thiết (Tốt nhất trên Google Colab có GPU):
```bash
pip install jax[cuda12] mujoco
```

### Bước 2: Chạy thử nghiệm mô phỏng 1024 drone song song trên GPU:
```bash
python Graduation2/train_mjx_ppo.py --num-envs 1024 --level 6 --timesteps 1000000
```
