"""Đối chiếu bản vector hóa với bản vòng lặp gốc trên nhiều episode, cùng seed, cùng action.
Chạy: python test_env_equivalence.py
"""
import sys
import time
sys.path.insert(0, r'c:/KLTN/MuJoCo-Drones/Graduation')
sys.path.insert(0, r'c:/KLTN/MuJoCo-Drones')
import numpy as np
import drone_ppo_curriculum_env as m

N_EPISODES = 6
MAX_STEPS_PER_EP = 40
TOL = 1e-9

env = m.DronePPOCurriculumEnv(gui=False)

# Bọc _computeSubstepReward: ở MỖI substep so khoảng cách mới vs cũ, và so check_collision có/không truyền min_d
stats = {"substeps": 0, "max_dist_diff": 0.0, "collision_mismatch": 0, "calls_legacy": 0}
orig_reward = env._computeSubstepReward


def checked_reward():
    d_new = env._get_min_obstacle_distance()
    d_old = env._get_min_obstacle_distance_legacy()
    stats["substeps"] += 1
    stats["max_dist_diff"] = max(stats["max_dist_diff"], abs(d_new - d_old))
    # check_collision dùng min_d truyền vào phải trùng với tự tính lại
    if env.check_collision(min_d=d_new) != env.check_collision():
        stats["collision_mismatch"] += 1
    return orig_reward()


env._computeSubstepReward = checked_reward

rng = np.random.default_rng(0)
ok = True
for level in (0, 3, 5):
    env.set_level(level)
    for ep in range(N_EPISODES):
        env.reset(seed=100 + ep)
        for _ in range(MAX_STEPS_PER_EP):
            a = rng.uniform(-1, 1, size=3).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break

print(f"Tổng substep đã đối chiếu : {stats['substeps']}")
print(f"Sai lệch khoảng cách lớn nhất (mới vs cũ): {stats['max_dist_diff']:.3e}")
print(f"Số lần check_collision(min_d) khác check_collision(): {stats['collision_mismatch']}")

ok = stats["max_dist_diff"] < TOL and stats["collision_mismatch"] == 0
print("=> KET QUA: TUONG DUONG, KHONG SAI LECH" if ok else "=> KET QUA: CO SAI LECH, KHONG DUNG BAN MOI")

# Đo tốc độ sau khi tối ưu
env._computeSubstepReward = orig_reward
env.set_level(5)
env.reset(seed=2)
t = time.perf_counter()
n = 0
for i in range(10):
    o, r, te, tr, inf = env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    n += 1
    if te or tr:
        env.reset(seed=3 + i)
print(f"Tốc độ: {(time.perf_counter() - t) / n * 1000:.1f} ms/step (trước tối ưu: ~200 ms/step khi có profiler)")
