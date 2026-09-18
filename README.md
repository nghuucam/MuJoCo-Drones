# MJ-drones-gym

**MuJoCo-based multi-drone Gymnasium environments for single & multi-agent reinforcement learning, hierarchical control, and vision-based obstacle avoidance.**

High-fidelity quadcopter simulation with GPU-vectorized environments, Dryden wind turbulence, domain randomization, obstacle generation, hierarchical PID/RL control, and vision-based navigation — all built on [MuJoCo](https://mujoco.org/).

---

## ✨ Features

- **MuJoCo physics** — faster, cleaner, and more numerically stable than PyBullet.
- **Gymnasium API** — drop-in compatible with Stable-Baselines3, CleanRL, RLlib, etc.
- **Hierarchical Control Architecture**:
  - **High-Level RL Agent (PPO)**: Computes spherical displacement commands $(\alpha, \beta, d)$ from FPV visual observations.
  - **Low-Level Controller (DSLPIDControl)**: High-rate inner-loop cascaded PID stabilizing 6-DoF attitude and motor RPMs.
- **Vision-based Obstacle Avoidance (`Graduation/`)**:
  - First-Person View (FPV) RGB camera observation ($64 \times 64 \times 3$).
  - Cylindrical & box obstacle fields with randomized heights ($0.1\text{m} - 3.0\text{m}$).
  - Spherical action space with hard ground-safety constraints ($z \ge 0.4\text{m}$).
  - Multi-core parallel training (`SubprocVecEnv`) with real-time TensorBoard monitoring.
- **Multi-drone support** — $N$ arbitrary drones with aerodynamic effects (ground effect, drag, downwash).
- **Multiple action & observation types** — RPM, normalized thrust, velocity, PID waypoint, kinematics, and RGB camera.
- **PettingZoo multi-agent** — parallel environment wrapper for Multi-Agent RL (MARL).
- **Task environments** — Hover, velocity tracking, waypoint navigation, formation, racing, and vision obstacle avoidance.

---

## 📦 Installation

This project uses modern Python packaging via `pyproject.toml` (PEP 517/518/621).

### 1. Basic Installation
```bash
git clone <this-repo>
cd MuJoCo-Drones/
pip install -e .
```

### 2. Full Installation (RL + Obstacle Avoidance + TensorBoard)
To install with reinforcement learning (Stable-Baselines3, TensorBoard, Rich, tqdm) and visualization dependencies:
```bash
pip install -e ".[all]"
```
Or for RL only:
```bash
pip install -e ".[rl]"
```

### System Requirements:
- Python $\ge$ 3.8
- MuJoCo $\ge$ 3.0
- Gymnasium $\ge$ 0.29
- NumPy $\ge$ 1.21

---

## 🚀 Quick Start

### 1. Vision-based Obstacle Avoidance (Graduation Thesis)

Train an autonomous drone agent using multi-process PPO and FPV camera input:

```bash
cd Graduation/

# Train with 4 CPU workers in parallel
python train_parallel.py --num-cpu 4 --timesteps 500000 --batch-size 64

# Monitor live training in TensorBoard
tensorboard --logdir logs/ppo_parallel
```

Test the environment and visual GUI:
```bash
# Verify environment API compliance
python test_env.py

# Interactive 3D visualization in MuJoCo viewer
python test_gui.py
```

### 2. PID Control Example

```python
import numpy as np
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
{{ ... }}

print(f"Final position error: {np.linalg.norm(env.pos[0] - target):.4f} m")
env.close()
```

### 3. Multi-Agent RL (PettingZoo)

```python
from multi_drone_mujoco.envs.multi_agent_aviary import MultiAgentAviary

env = MultiAgentAviary(num_drones=3)
env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terms, truncs, infos = env.step(actions)
```

---

## 📂 Project Structure

```
MuJoCo-Drones/
├── pyproject.toml              # Modern package configuration & dependencies (PEP 621)
├── setup.py                    # Backward compatibility shim
├── README.md                   # Project documentation
│
├── Graduation/                 # Autonomous Navigation & Obstacle Avoidance
│   ├── drone_ppo_env.py        # Gymnasium env (FPV camera, spherical action, safety bound)
│   ├── config.py               # World, obstacle, and reward hyperparameters
│   ├── train_parallel.py       # Multi-core PPO training with SubprocVecEnv & TensorBoard
│   ├── train_ppo.py            # Single-process baseline training script
│   ├── test_env.py             # SB3 check_env & action-step verification
│   └── test_gui.py             # 3D interactive viewer with PID & obstacle field
│
├── multi_drone_mujoco/         # Core simulation library
│   ├── envs/                   # Gymnasium environments (Hover, Velocity, Race, etc.)
│   ├── control/                # Cascaded PID controllers (PIDControl, DSLPIDControl)
│   ├── utils/                  # Enums, logger, coordinate transformations
│   └── examples/               # Reference scripts
└── tests/                      # Unit and integration tests
```

---

## 🧪 Running Tests

```bash
pytest multi_drone_mujoco/tests/ -v
```

---

## 📜 Citation & Acknowledgements

- **MuJoCo Menagerie**: Bitcraze Crazyflie 2.x MJCF model.
- **gym-pybullet-drones**: Reference architecture for quadcopter gym environments.
- **Stable-Baselines3**: PPO algorithm and vectorization wrappers.

---

## 📄 License

MIT License.
