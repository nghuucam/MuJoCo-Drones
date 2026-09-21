# MuJoCo-Drones (MJ-drones-gym)

**High-fidelity MuJoCo-based Gymnasium and PettingZoo environments for single & multi-quadcopter reinforcement learning (RL), hierarchical control, aerodynamic wind disturbances, and vision-based obstacle avoidance.**

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Conda Environment & Installation](#-conda-environment--installation)
  - [1. Create Conda Environment (`conda create`)](#1-create-conda-environment-conda-create)
  - [2. Install PyTorch with CUDA / CPU](#2-install-pytorch-with-cuda--cpu)
  - [3. Install MuJoCo-Drones Package](#3-install-mujoco-drones-package)
  - [4. Verify Installation](#4-verify-installation)
- [Reinforcement Learning with Stable-Baselines3 (SB3)](#-reinforcement-learning-with-stable-baselines3-sb3)
  - [Supported RL Algorithms](#supported-rl-algorithms)
  - [Policy Architectures & Observation Spaces](#policy-architectures--observation-spaces)
  - [Multi-Core Parallel Vectorized Training (`SubprocVecEnv`)](#multi-core-parallel-vectorized-training-subprocvecenv)
  - [Ready-to-Run SB3 Training Scripts](#ready-to-run-sb3-training-scripts)
  - [Environment Validation (`check_env`)](#environment-validation-check_env)
  - [Evaluation, Inference & Policy Visualization](#evaluation-inference--policy-visualization)
  - [TensorBoard Monitoring & Key Metrics](#tensorboard-monitoring--key-metrics)
  - [Callbacks & Model Checkpointing](#callbacks--model-checkpointing)
- [Hierarchical Control Architecture](#-hierarchical-control-architecture)
- [Available Gymnasium Environments](#-available-gymnasium-environments)
- [Physics Wrappers & Domain Randomization](#-physics-wrappers--domain-randomization)
- [Interactive 3D Demo Suites](#-interactive-3d-demo-suites)
- [Troubleshooting & Windows Notes](#-troubleshooting--windows-notes)
- [License & Citation](#-license--citation)

---

## ✨ Key Features

- **MuJoCo Physics Core**: High-speed numerical integration (240Hz physics sub-stepping) with realistic mass matrices, gyroscopic precession, aerodynamics, ground effect, and downwash.
- **Full Gymnasium & PettingZoo Compatibility**: Seamless plug-and-play integration with Stable-Baselines3 (SB3), CleanRL, Ray RLlib, and PettingZoo MARL.
- **Deep Stable-Baselines3 (SB3) Integration**:
  - Vectorized multi-process rollouts (`SubprocVecEnv`, `VecMonitor`).
  - Pre-configured PPO and SAC pipelines with `MlpPolicy` (kinematics) and `CnnPolicy` (FPV vision).
  - Robust evaluation callbacks, checkpointing, and real-time TensorBoard logging.
- **Hierarchical Control Architecture**:
  - **High-level RL Agent**: Computes target waypoints, linear velocities, or spherical displacement commands.
  - **Low-level Controller (`DSLPIDControl` / `PIDControl`)**: Cascaded attitude-rate and motor RPM control running at high frequency.
- **Vision-Based Obstacle Avoidance (`Graduation/`)**:
  - First-Person View (FPV) RGB camera input ($64 \times 64 \times 3$).
  - Procedural obstacle fields (cylinders, boxes, randomized heights).
  - Spherical action spaces with ground-safety constraints.
- **Aerodynamic Disturbance Modeling (`WindWrapper`)**:
  - Constant wind fields, stochastic discrete gusts, MIL-F-8785C Dryden turbulence, sinusoidal oscillating winds, and combined extreme storm weather.
  - Compliant (`--soft-pid`) vs Stiff PID modes for realistic vehicle rocking and swaying.
- **Procedural Environments (`ObstaclesWrapper`)**:
  - Themed obstacle field generation: `FOREST`, `URBAN`, `INDOOR`, `RANDOM`, `GATES`.

---

## 🐍 Conda Environment & Installation

Follow these step-by-step instructions to create an isolated Python virtual environment using Anaconda or Miniconda.

### 1. Create Conda Environment (`conda create`)

Open terminal (Anaconda Prompt / PowerShell on Windows, or bash on Linux/macOS):

```bash
# Create a dedicated conda environment with Python 3.10
conda create -n mujoco_drones python=3.10 -y

# Activate the newly created environment
conda activate mujoco_drones
```

> **Note**: Python 3.10 or 3.11 is strongly recommended for maximum compatibility with MuJoCo, PyTorch, and Stable-Baselines3.

---

### 2. Install PyTorch with CUDA / CPU

Install PyTorch according to your hardware setup before installing the repository dependencies:

#### Option A: With NVIDIA GPU (CUDA 12.1 Acceleration — Recommended for SB3 CNN Training)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CPU Only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### 3. Install MuJoCo-Drones Package

Clone the repository and install in editable mode (`-e`):

```bash
# Clone the repository
git clone https://github.com/nghuucam/MuJoCo-Drones.git
cd MuJoCo-Drones

# Upgrade pip and packaging tools
pip install --upgrade pip setuptools wheel

# Install with ALL reinforcement learning and visualization dependencies
pip install -e ".[all]"
```

If you only need specific feature groups, you can choose:
```bash
# RL dependencies only (Stable-Baselines3, TensorBoard, tqdm, rich)
pip install -e ".[rl]"

# Multi-Agent RL only (PettingZoo)
pip install -e ".[marl]"

# Visualization tools only (matplotlib, Pillow)
pip install -e ".[viz]"
```

---

### 4. Verify Installation

Run this quick one-liner to verify that MuJoCo, Gymnasium, and Stable-Baselines3 are correctly linked:

```bash
python -c "import mujoco, gymnasium, stable_baselines3, multi_drone_mujoco; print('All core modules imported successfully!')"
```

---

## 🤖 Reinforcement Learning with Stable-Baselines3 (SB3)

[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/) is the primary reinforcement learning framework integrated into MuJoCo-Drones.

### Supported RL Algorithms

| Algorithm | Type | Observation Space | Action Space | Typical Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **PPO** (*Proximal Policy Optimization*) | On-Policy | Kinematics (`MlpPolicy`) or FPV RGB (`CnnPolicy`) | Continuous (`Box`) | Hovering, Waypoint tracking, Obstacle avoidance, Gate navigation |
| **SAC** (*Soft Actor-Critic*) | Off-Policy | Kinematics (`MlpPolicy`) | Continuous (`Box`) | Sample-efficient continuous torque/RPM control |

---

### Policy Architectures & Observation Spaces

#### 1. Kinematic States (`MlpPolicy`)
Used in environments like `HoverAviary`, `VelocityAviary`, `RaceAviary`, and `FormationAviary`:
- **Observation**: 12-DoF or 18-DoF kinematic vector:
  $$\mathbf{o} = [x, y, z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, \omega_x, \omega_y, \omega_z, \Delta x, \Delta y, \Delta z]$$
- **Network**: Multi-Layer Perceptron (MLP) with 2 hidden layers of 64 or 256 units (`Tanh` or `ReLU`).

#### 2. Visual Camera Input (`CnnPolicy`)
Used in `Graduation/drone_ppo_env.py` for autonomous obstacle avoidance:
- **Observation**: $64 \times 64 \times 3$ uint8 RGB image captured from the onboard drone FPV camera.
- **Network**: 2D Convolutional Neural Network (Nature CNN architecture) feature extractor feeding into Actor-Critic heads.

---

### Multi-Core Parallel Vectorized Training (`SubprocVecEnv`)

MuJoCo's C-based physics engine is exceptionally fast. By wrapping environments in SB3's `SubprocVecEnv`, you run parallel physics simulations across multiple CPU cores, achieving hundreds of thousands of simulation steps per minute.

```python
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from multi_drone_mujoco.envs.hover_aviary import HoverAviary

def make_env(rank: int, seed: int = 0):
    def _init():
        env = HoverAviary(gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    num_cpus = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])
    vec_env = VecMonitor(vec_env, filename="./logs/monitor.csv")

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs/tb")
    model.learn(total_timesteps=200_000)
    model.save("ppo_hover_model")
    vec_env.close()
```

---

### Ready-to-Run SB3 Training Scripts

MuJoCo-Drones comes with pre-configured training scripts for quick benchmarking:

#### 1. Single-Drone Hover Baseline
A fast training script to test PPO on position holding:
```bash
python multi_drone_mujoco/examples/train_rl.py
```

#### 2. Train Across Multiple Aviary Environments
Trains PPO or SAC sequentially across `HoverAviary`, `VelocityAviary`, `FlyThroughAviary`, `RaceAviary`, and `FormationAviary`:
```bash
python multi_drone_mujoco/examples/train_all_envs.py
```

#### 3. Configurable Single & Multi-Drone PPO
Supports custom step counts, TensorBoard logging, and multi-drone centralized control:
```bash
# Train single-drone hover (100,000 timesteps)
python -m multi_drone_mujoco.examples.learn

# Train multi-drone hover with 2 drones
python -m multi_drone_mujoco.examples.learn --multiagent true
```

#### 4. Vision-Based Obstacle Avoidance (`Graduation/`)
Multi-worker parallel PPO training from raw FPV camera visual observations:
```bash
cd Graduation/

# Train with 4 CPU workers in parallel for 500,000 timesteps
python train_parallel.py --num-cpu 4 --timesteps 500000 --batch-size 64 --lr 3e-4

# Single-worker baseline training
python train_ppo.py
```

---

### Environment Validation (`check_env`)

Before training, verify that your custom environment strictly conforms to Gymnasium/SB3 specifications:
```bash
cd Graduation/
python test_env.py
```

---

### Evaluation, Inference & Policy Visualization

#### Visualize Trained Policy in 3D
Watch the trained policy navigate in real-time 3D:
```bash
# Visualize trained hover policy
python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip

# Visualize vision-based obstacle avoidance agent
cd Graduation/
python test_gui.py
```

---

### TensorBoard Monitoring & Key Metrics

Monitor real-time training curves, reward progression, policy entropy, and value function loss:

```bash
# Launch TensorBoard server
tensorboard --logdir logs/
```
Open your browser at `http://localhost:6006/`.

#### Critical Metrics to Inspect:
- `rollout/ep_rew_mean`: Mean episode reward (should steadily increase toward target threshold).
- `rollout/ep_len_mean`: Mean episode duration (longer duration indicates fewer crashes).
- `train/approx_kl`: Approximate Kullback-Leibler divergence (monitors step size stability in PPO).
- `train/entropy_loss`: Policy exploration entropy (decreases as policy converges).
- `train/value_loss`: Mean squared error of the critic value estimation.
- `train/explained_variance`: How well the value function predicts returns (ideally $> 0.8$).

---

### Callbacks & Model Checkpointing

All training scripts utilize SB3 callbacks to prevent data loss during long runs:
- **`CheckpointCallback`**: Saves periodic checkpoints (e.g. every 10,000 steps).
- **`EvalCallback`**: Evaluates policy on a separate un-vectorized environment, logging deterministic performance and saving `best_model.zip`.

---

## 🕹️ Hierarchical Control Architecture

MuJoCo-Drones supports hierarchical control combining reinforcement learning with classic control theory:

```
┌────────────────────────────────────────────────────────┐
│               High-Level Policy (PPO)                  │
│       Input: Visual FPV Camera (64x64x3) or State       │
│       Output: Spherical displacement commands (α, β, d) │
└───────────────────────────┬────────────────────────────┘
                            │ (Control frequency: 8-48 Hz)
                            ▼
┌────────────────────────────────────────────────────────┐
│          Low-Level Cascaded Controller (PID)           │
│           (DSLPIDControl / PIDControl)                 │
│       Input: Target position & attitude setpoints       │
│       Output: Individual motor RPMs (4 rotors)         │
└───────────────────────────┬────────────────────────────┘
                            │ (Physics frequency: 240 Hz)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 MuJoCo Physics Engine                  │
│       xfrc_applied / qfrc_applied multi-substep        │
└───────────────────────────┬────────────────────────────┘
```

This decoupled architecture allows the high-level RL agent to focus on path planning and spatial obstacle avoidance, while the deterministic PID controller handles fast 6-DoF attitude stabilization.

---

## 🌍 Available Gymnasium Environments

| Environment Class | Module | Drones | Action Space | Observation Space | Task Objective |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`HoverAviary`** | `envs.hover_aviary` | 1 | Motor RPM / Normalized | 12D Kinematics | Hover at $[0, 0, 1.0]\text{m}$ |
| **`VelocityAviary`** | `envs.velocity_aviary` | 1 | Linear Velocity $(v_x, v_y, v_z)$ | 12D Kinematics | Track continuous target velocity vector |
| **`FlyThroughAviary`** | `envs.fly_through_aviary` | 1 | Motor RPM / Waypoint | Kinematics + Gate relative pose | Fly through a square/circular gate |
| **`RaceAviary`** | `envs.race_aviary` | $1+$ | Motor RPM / Waypoint | Kinematics + Multi-gate waypoint | High-speed multi-gate obstacle course |
| **`MultiHoverAviary`** | `envs.multi_hover_aviary` | $N$ | Centralized Motor RPMs | Concatenated $N \times 12$D | Simultaneous multi-drone hover |
| **`FormationAviary`** | `envs.formation_aviary` | $N$ | Centralized Motor RPMs | Concatenated $N \times 12$D | Maintain geometric formation pattern |
| **`MultiAgentAviary`** | `envs.multi_agent_aviary` | $N$ | Decentralized per-drone | Decentralized per-drone | PettingZoo ParallelEnv MARL |
| **`DronePPOEnv`** | `Graduation.drone_ppo_env` | 1 | Spherical $(\alpha, \beta, d)$ | $64 \times 64 \times 3$ RGB FPV | FPV visual obstacle avoidance |

---

## 🌪️ Physics Wrappers & Domain Randomization

Located in `multi_drone_mujoco/wrappers/`:

### 1. `WindWrapper`
Injects aerodynamic wind disturbances directly into MuJoCo external force buffers:
- **`CONSTANT`**: Constant directional wind vector.
- **`GUST`**: Stochastic discrete wind impulses.
- **`DRYDEN`**: Continuous military standard (MIL-F-8785C) atmospheric turbulence.
- **`SINUSOIDAL`**: Periodic harmonic oscillating wind.
- **`COMBINED`**: Extreme storm weather superposition.

### 2. `ObstaclesWrapper`
Generates procedural obstacles with collision checking:
- Themes: `FOREST`, `URBAN`, `INDOOR`, `RANDOM`, `GATES`.

### 3. `CurriculumWrapper`
Dynamically increases environment difficulty (e.g. higher wind speeds, denser obstacles) as agent evaluation return crosses milestones.

---

## 🖥️ Interactive 3D Demo Suites

The `Test/` directory contains comprehensive test suites and interactive 3D visualizations:

### 1. Core Aviary Demos (`Test/aviary_demos/`)
```bash
cd Test/aviary_demos
python 01_test_hover.py --gui
python 02_test_velocity.py --gui
python 03_test_fly_through_gate.py --gui
python 04_test_race_track.py --gui
python 05_test_multi_hover.py --gui
python 06_test_formation.py --gui
python 07_test_pettingzoo_marl.py --gui
```

### 2. Procedural Obstacles Demos (`Test/aviary_demos2/`)
```bash
cd Test/aviary_demos2
python 01_test_obstacles_forest.py --gui
python 02_test_obstacles_urban.py --gui
python 03_test_obstacles_indoor.py --gui
python 04_test_obstacles_random.py --gui
python 05_test_obstacles_gates.py --gui
python 06_compare_obstacle_density.py --gui
python run_all_obstacles.py --gui
```

### 3. Aerodynamic Wind Demos (`Test/aviary_demos3/`)
```bash
cd Test/aviary_demos3
# Test Dryden turbulence with compliant PID (visible drone rocking/swaying)
python 03_test_wind_dryden.py --gui --soft-pid

# Test animated 3D aerodynamic wind gust swirl streamlines
python 08_test_wind_swirls_icon.py --gui --soft-pid

# Test visible wind drift & tilt
python 07_test_wind_drift_visible.py --gui --mode resist
python 07_test_wind_drift_visible.py --gui --mode drift

# Run all wind suites
python run_all_wind.py --gui --soft-pid
```

---

## 🔧 Troubleshooting & Windows Notes

### 1. Multiprocessing on Windows (`freeze_support`)
On Windows, Python uses `spawn` instead of `fork`. Always wrap your script entry points with:
```python
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    main()
```

### 2. Headless Server Rendering (Linux vs Windows)
- **Windows**: Native interactive 3D rendering uses standard OpenGL through GLFW.
- **Linux Headless (Cloud/Cluster)**: Use EGL for offscreen camera rendering without an active display:
  ```bash
  export MUJOCO_GL=egl
  ```

### 3. Check GPU Availability in PyTorch
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 📜 License & Citation

This project is licensed under the [MIT License](LICENSE).

If you use MuJoCo-Drones in your research, please cite:
```bibtex
@misc{mujoco_drones_2026,
  author = {Nghuu Cam and Contributors},
  title = {MuJoCo-Drones: High-Fidelity Multi-Quadcopter Gymnasium Environments for Reinforcement Learning and Hierarchical Control},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/nghuucam/MuJoCo-Drones}}
}
```
