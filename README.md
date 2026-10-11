# AUTONOMOUS QUADROTOR NAVIGATION & 3D OBSTACLE AVOIDANCE USING PPO & CURRICULUM LEARNING

> **Graduation Thesis Project:** Autonomous Quadrotor 3D Obstacle Avoidance via Proximal Policy Optimization (PPO), Curriculum Learning, and Multimodal Sensory Fusion (64x64 FPV Camera + Kinematic State Vector + APF Guidance) in MuJoCo Physics Engine.

---

## 📑 TABLE OF CONTENTS
1. [Project Overview: OldCode vs. Graduation Modules](#1-project-overview-oldcode-vs-graduation-modules)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Environment Setup with Anaconda](#3-environment-setup-with-anaconda)
4. [Verification of `pyproject.toml` Dependencies](#4-verification-of-pyprojecttoml-dependencies)
5. [Detailed Guide to `Graduation/` Scripts, Parameters & Execution](#5-detailed-guide-to-graduation-scripts-parameters--execution)
   - [5.1. `train_parallel_curriculum.py` (Multi-Worker Parallel Training)](#51-train_parallel_curriculumpy---parallel-multi-processing-training)
   - [5.2. `train_ppo_curriculum.py` (Single-Worker Debugging)](#52-train_ppo_curriculumpy---single-worker-debugging)
   - [5.3. `tune_optuna.py` (Automated Hyperparameter Optimization)](#53-tune_optunapy---automated-hyperparameter-tuning-via-optuna)
   - [5.4. `eval_model.py` (Model Evaluation & Failure Recording)](#54-eval_modelpy---3d-visual-evaluation--failure-buffer-logger)
   - [5.5. `plot_failed_trajectories.py` (3D/2D Failed Trajectory Visualization)](#55-plot_failed_trajectoriespy---failed-trajectory-plotter)
   - [5.6. `drone_ppo_curriculum_env.py` (Core Gymnasium Environment)](#56-drone_ppo_curriculum_envpy---core-gymnasium-environment)
   - [5.7. `config.py` (Global Parameters & Map Layout)](#57-configpy---system-configurations--curriculum-ranges)
   - [5.8. `analyze_flight_log.py` (Flight Log CSV Analytics)](#58-analyze_flight_logpy---flight-log-analytics)
6. [Streamlined Reward Function Specification](#6-streamlined-reward-function-specification)

---

## 1. Project Overview: OldCode vs. Graduation Modules

This repository presents an end-to-end research and engineering framework for autonomous quadrotor navigation and obstacle avoidance in cluttered 3D dynamic environments:

* **📁 `OldCode/` (Legacy Phase - Baseline & Motivation for Upgrade):**
  * Represents the initial phase of the research, implementing **DQN (Deep Q-Network)** and **D3QN (Dueling Double DQN)** algorithms.
  * *Limitations of Legacy Code:* Discrete action spaces (limited fixed directional steps), oversimplified collision models, absence of onboard vision (FPV camera), and lack of continuous aerodynamic attitude control.
  * *Archival Purpose:* Preserved as baseline benchmarks and empirical evidence demonstrating the evolutionary progress of the research for the final graduation defense.

* **📁 `Graduation/` (Current Production Phase - Core Thesis Focus):**
  * Full transition to **PPO (Proximal Policy Optimization)** with **Continuous Action Spaces** formulated in the Aviation Body Frame ($\alpha, \beta, d$).
  * Implementation of **Curriculum Learning (7 Progressive Difficulty Levels)** scaling the target range from 6m to 21m and increasing procedurally generated spiked obstacles from 0 to 15 pillars.
  * Multimodal sensory fusion integrating **64x64 RGB FPV camera feeds** (CNN feature extractor), **18-dimensional kinematic telemetry** (MLP policy), and **Artificial Potential Field (APF) guidance hints**.

---

## 2. Project Directory Structure

```
c:\KLTN\MuJoCo-Drones\
│
├── README.md                           # Comprehensive documentation and operational guide
├── pyproject.toml                      # PEP 621 package metadata & complete dependency tree
├── setup.py                            # Backward compatibility for editable pip installation
│
├── Graduation/                         # CORE GRADUATION THESIS MODULE (LATEST PRODUCTION)
│   ├── config.py                       # Global configurations, map limits, FPV camera & levels
│   ├── drone_ppo_curriculum_env.py     # Gymnasium environment (MuJoCo C-API + FPV + Spiked Pillars)
│   ├── train_parallel_curriculum.py    # Multi-processing parallel training (SubprocVecEnv)
│   ├── train_ppo_curriculum.py         # Single-worker training designed for debugging
│   ├── tune_optuna.py                  # Automated PPO hyperparameter tuning (Bayesian TPE)
│   ├── eval_model.py                   # Model evaluation, telemetry monitor & failed trajectory logger
│   ├── plot_failed_trajectories.py     # 3D and 2D top-down failed trajectory plotter
│   ├── flight_logger_callback.py       # Custom SB3 callback logging flight telemetry to CSV
│   ├── analyze_flight_log.py           # Post-training CSV analytics, reward & win-rate curve plotting
│   ├── models/                         # Checkpoint weights and final exported policies (.zip)
│   ├── logs/                           # TensorBoard event files, CSV logs & trajectory figures
│   └── optuna_results/                 # SQLite database & interactive HTML Plotly reports
│
├── OldCode/                            # LEGACY CODEBASE (FOR BENCHMARKING & PROGRESS COMPARISON)
│   ├── DQN/                            # DQN implementation (Single, Parallel, Optimize, Analyze)
│   │   ├── Single/                     # Single-worker DQN training
│   │   ├── Parallel/                   # Parallel DQN training
│   │   ├── Optimize/                   # DQN optimization scripts
│   │   └── Analyze/                    # Legacy DQN analysis scripts & plots
│   └── D3QN/                           # Dueling Double DQN implementation
│       ├── Single/                     # Single-worker D3QN training
│       ├── Parallel/                   # Parallel D3QN training
│       ├── Optimize/                   # D3QN optimization scripts
│       └── Analyze/                    # Legacy D3QN analysis scripts & plots
│
└── multi_drone_mujoco/                 # UNDERLYING DRONE SIMULATION ENGINE
    ├── assets/cf2/                     # Official Crazyflie 2.X 3D meshes (.obj) & physical attributes
    ├── control/dsl_pid_control.py      # Low-level 48Hz DSL-PID position & attitude controller
    └── envs/base_aviary.py             # Gymnasium interface wrapping MuJoCo physics bindings
```

---

## 3. Environment Setup with Anaconda

### Step 1: Create the Conda Virtual Environment
Open **Anaconda Prompt** or **PowerShell** and execute:

```bash
# 1. Create a virtual environment with Python 3.10
conda create -n mujoco_drone python=3.10 -y

# 2. Activate the environment
conda activate mujoco_drone
```

### Step 2: Install PyTorch with CUDA Acceleration
For systems equipped with an NVIDIA GPU (e.g., NVIDIA GeForce RTX 3050 Laptop GPU):

```bash
# Install PyTorch with CUDA 12.1 runtime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

*Verify CUDA installation:*
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 3: Install All Project Dependencies from `pyproject.toml`
Navigate to the root repository folder and install in editable mode:

```bash
cd c:\KLTN\MuJoCo-Drones
pip install -e .
```

---

## 4. Verification of `pyproject.toml` Dependencies

The [`pyproject.toml`](file:///c:/KLTN/MuJoCo-Drones/pyproject.toml) configuration has been thoroughly verified and includes **100% of the required libraries** for training, hyperparameter tuning, simulation, and data analytics:

| Category | Library in `pyproject.toml` | Version | Purpose |
| :--- | :--- | :---: | :--- |
| **Physics & RL** | `gymnasium`<br>`mujoco`<br>`stable-baselines3` | $\ge 0.29.0$<br>$\ge 3.0.0$<br>$\ge 2.0.0$ | Standard RL interface, native MuJoCo bindings, vectorized PPO. |
| **Deep Learning** | `torch` | $\ge 2.0.0$ | Multimodal Actor-Critic neural networks (CNN + MLP). |
| **Computer Vision** | `Pillow` | $\ge 9.0.0$ | FPV camera frame rendering and array preprocessing. |
| **Math & Data** | `numpy`<br>`scipy`<br>`pandas` | $\ge 1.21.0$<br>$\ge 1.9.0$<br>$\ge 1.3.0$ | Linear algebra, spatial transformations, tabular flight logging. |
| **Visualization** | `matplotlib`<br>`seaborn`<br>`tensorboard` | $\ge 3.5.0$<br>$\ge 0.11.0$<br>$\ge 2.10.0$ | 3D/2D trajectory plotting, real-time loss/reward telemetry tracking. |
| **Optimization** | `optuna`<br>`plotly` | $\ge 3.5.0$<br>$\ge 5.15.0$ | Bayesian TPE hyperparameter optimization & interactive HTML graphs. |
| **Utilities** | `tqdm` | $\ge 4.60.0$ | Console progress tracking. |

---

## 5. Detailed Guide to `Graduation/` Scripts, Parameters & Execution

All command lines below should be executed from within the `Graduation/` directory:
```bash
cd c:\KLTN\MuJoCo-Drones\Graduation
```

---

### 5.1. `train_parallel_curriculum.py` — Parallel Multi-Processing Training

* **Function:** Main training pipeline. Initializes `SubprocVecEnv` across multiple CPU workers running parallel MuJoCo simulations, coupled with GPU acceleration for PPO optimization. Features experiment isolation via `--exp-name` and checkpoint resumption via `--resume-model`.
* **CLI Parameter Reference:**

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--exp-name` | `str` | `None` | **Experiment name for complete isolation.** Creates isolated subdirectories for models (`models/<exp_name>/`), TensorBoard logs (`logs/<exp_name>/`), and flight logs (`drone_flight_log_<exp_name>.csv`). Prevents overwriting across concurrent runs. |
| `--num-cpu` | `int` | `4` | Number of parallel CPU simulation workers (recommended: 4 to 6). |
| `--timesteps` | `int` | `3000000` | Total environment training timesteps. |
| `--save-freq` | `int` | `500000` | Periodic checkpoint saving interval (in global timesteps). |
| `--resume-model` | `str` | `None` | Path to an existing `.zip` model for continued training (Fine-tuning). |
| `--lr` | `float` | `1e-4` | Learning rate for Adam optimizer. |
| `--n-steps` | `int` | `512` | Rollout steps per worker before update ($512 \times 6 = 3072$ buffer size). |
| `--batch-size` | `int` | `256` | Mini-batch size processed by GPU during gradient descent. |
| `--n-epochs` | `int` | `10` | Optimization epochs per update cycle. |
| `--clip-range` | `float` | `0.2` | PPO surrogate clipping threshold ($1 - \epsilon$ to $1 + \epsilon$). |
| `--ent-coef` | `float` | `0.001` | Entropy coefficient encouraging action space exploration. |
| `--gamma` | `float` | `0.99` | Future reward discount factor. |
| `--gae-lambda` | `float` | `0.95` | Generalized Advantage Estimation factor. |
| `--target-kl` | `float` | `0.05` | Target KL divergence threshold for early stopping. |
| `--device` | `str` | `auto` | Computational backend: `auto`, `cuda`, or `cpu`. |
| `--no-curriculum` | `flag` | `False` | **Disables curriculum learning**, locking the environment at maximum difficulty (Level 6). |
| `--start-level` | `int` | `0` | Starting difficulty level (0 to 6). |

* **Execution Examples:**

```bash
# 1. Full training with Curriculum Learning from scratch (6 CPUs, isolated experiment name 'run_curr'):
python train_parallel_curriculum.py --exp-name run_curr --num-cpu 6 --timesteps 3000000 --save-freq 500000

# 2. Fine-tuning an existing model on Level 6 only with Optuna-tuned hyperparameters:
python train_parallel_curriculum.py --exp-name fine_tune_level6 \
  --resume-model models/run_curr/drone_ppo_run_curr_final.zip \
  --start-level 6 --no-curriculum \
  --lr 6e-5 --n-epochs 8 --clip-range 0.15 --ent-coef 0.0002 --gamma 0.98 \
  --num-cpu 6 --timesteps 1000000
```

---

### 5.2. `train_ppo_curriculum.py` — Single-Worker Debugging

* **Function:** Runs the PPO training pipeline within a single worker environment. Ideal for line-by-line debugging, verifying controller physics, or running on systems with limited CPU core counts.
* **Execution:**
```bash
python train_ppo_curriculum.py --timesteps 500000 --lr 1e-4 --batch-size 64 --start-level 0
```

---

### 5.3. `tune_optuna.py` — Automated Hyperparameter Tuning via Optuna

* **Function:** Utilizes Bayesian Tree-structured Parzen Estimator (TPE) sampling to identify optimal PPO hyperparameters (`learning_rate`, `n_epochs`, `clip_range`, `ent_coef`, `gamma`, `gae_lambda`). Implements **MedianPruner** to terminate unpromising trials early, and supports warm-starting from pre-trained checkpoints via `--resume-model`.
* **CLI Parameter Reference:**

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--resume-model` | `None` | Path to pre-trained model to optimize parameters directly on Level 6. |
| `--start-level` | `0` | Difficulty level for tuning trials (recommended: `6`). |
| `--n-trials` | `15` | Total number of optimization trials. |
| `--trial-timesteps`| `100000` | Sample timesteps per trial (20,000 to 50,000 for rapid exploration). |
| `--num-cpu` | `4` | Parallel CPU workers allocated per trial. |
| `--study-name` | `drone_ppo_optuna_study` | Name of the SQLite-backed study. |

* **Execution:**
```bash
python tune_optuna.py --resume-model models/drone_ppo_curriculum_ver4_final.zip --start-level 6 --num-cpu 4 --n-trials 10 --trial-timesteps 30000
```
*Outputs are saved to `optuna_results/optuna_study.db`, alongside interactive HTML plots (`param_importances.html`, `parallel_coordinate.html`, `optimization_history.html`).*

---

### 5.4. `eval_model.py` — 3D Visual Evaluation & Failure Buffer Logger

* **Function:** Loads trained policy weights to evaluate flight performance over $N$ episodes. Features real-time 3D MuJoCo rendering, FPV camera display, comprehensive failure categorizations (Collision, Over-map, Timeout), and **automatically saves complete trajectory data of all failed episodes** to a JSON file.
* **CLI Parameter Reference:**

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--model-path` | Auto-detect | Path to target `.zip` model file. |
| `--level` | `6` | Test curriculum level (0 to 6). |
| `--episodes` | `5` | Number of evaluation episodes. |
| `--no-gui` | `False` | Disables 3D GUI for high-throughput headless testing. |
| `--quiet` | `False` | Suppresses per-step telemetry console tables. |
| `--failed-log` | `logs/failed_trajectories.json` | Path to export failed trajectory logs. |

* **Execution Examples:**

```bash
# Interactive evaluation of 10 episodes on Level 6 with 3D GUI:
python eval_model.py --model-path models/drone_ppo_curriculum_ver4_final.zip --level 6 --episodes 10

# High-throughput headless evaluation (50 episodes) to record failure trajectories:
python eval_model.py --model-path models/drone_ppo_curriculum_ver4_final.zip --level 6 --episodes 50 --no-gui --quiet
```

---

### 5.5. `plot_failed_trajectories.py` — Failed Trajectory Plotter

* **Function:** Reads `failed_trajectories.json` generated by `eval_model.py` and produces publication-quality dual-pane visual analytics for each failed flight:
  * **Left Subplot (3D Space):** Illustrates pillar cylinders, protruding spikes, Start coordinate, Goal coordinate, drone flight trajectory, and the exact collision point.
  * **Right Subplot (2D Top-Down XY View):** Highlights obstacle clearance, horizontal evasion paths, step annotations, and the exact termination location.
* **CLI Parameter Reference:**

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--input` | `logs/failed_trajectories.json` | Path to input JSON file. |
| `--output-dir` | `logs/failed_plots/` | Directory where `.png` figures are saved. |
| `--show` | `False` | Opens an interactive Matplotlib 3D viewer. |

* **Execution:**
```bash
python plot_failed_trajectories.py --input logs/failed_trajectories.json --show
```

---

### 5.6. `drone_ppo_curriculum_env.py` — Core Gymnasium Environment

* **Function:** Implements the core simulation environment:
  * **Observation Space:** Multimodal `Dict({"rgb": Box(64,64,3), "state": Box(18,)})`.
  * **Action Space:** Continuous `Box(3,)` defining $[\alpha, \beta, d]$ in the Body Frame.
  * **Procedural Spiked Pillars:** MuJoCo XML generated on reset with randomized heights and spike distributions.
  * **Guidance Fields:** Artificial Potential Field (APF) calculation providing real-time attractive/repulsive yaw & pitch cues.

---

### 5.7. `config.py` — System Configurations & Curriculum Ranges

* **Function:** Central repository for global constants and hardware simulation limits:
  * `GOAL_Y_RANGES`: Target distance ranges per level (Level 0: $[-6, -5]$ up to Level 6: $[-21, -19]$).
  * `OBSTACLE_COUNTS`: Obstacle count scaling per level (Level 0: 0 up to Level 6: 15 pillars).
  * `MAX_STEPS`: Episode step truncation ceiling (`140` steps).
  * `COLLISION_MARGIN`: High-precision obstacle proximity margin (`0.05m`).
  * `IMG_WIDTH`, `IMG_HEIGHT`: FPV camera rendering resolution (`64x64`).

---

### 5.8. `analyze_flight_log.py` — Flight Log Analytics

* **Function:** Parses flight logs (`drone_flight_log_parallel_ver4.csv`) generated during training to compute success rates, collision distributions, and average rewards across curriculum levels, plotting cumulative learning curves.
* **Execution:**
```bash
python analyze_flight_log.py --log logs/drone_flight_log_parallel_ver4.csv
```

---

## 6. Streamlined Reward Function Specification

The reward function in [`drone_ppo_curriculum_env.py`](file:///c:/KLTN/MuJoCo-Drones/Graduation/drone_ppo_curriculum_env.py) has been refined to eliminate counterproductive constraints (removing the 3x regression penalty, proximity milestone bait, and velocity heading restrictions) to provide the agent with unconstrained behavioral freedom for smooth obstacle avoidance maneuvers:

$$\mathcal{R}_{\text{substep}} = R_{\text{time}} + R_{\text{progress}} + R_{\text{idle}} + R_{\text{proximity}} + R_{\text{altitude}} + R_{\text{terminal}}$$

1. **$R_{\text{time}} = -0.002$:** Per-substep time expenditure motivating the drone to advance toward the destination efficiently.
2. **$R_{\text{progress}} = +1.0 \times \Delta d_{\text{best}}:$** Distance advancement reward granted when breaking the record minimum distance to the target.
3. **$R_{\text{idle}} = -0.005:$** Loitering penalty triggered only when the flight corridor ahead is clear ($>1.5\text{m}$) but velocity remains below $0.1\text{m/s}$.
4. **$R_{\text{proximity}} = -0.005:$** Hazard warning penalty activated when drone distance to any pillar or spike surface falls below $0.1\text{m}$.
5. **$R_{\text{altitude}} = -0.01:$** Boundary penalty applied when grazing the floor envelope ($z \le 0.85\text{m}$) or ceiling limit ($z \ge 2.45\text{m}$).
6. **$R_{\text{terminal}}:$** 
   * **Goal Reached ($Win$):** $+100.0$ points.
   * **Collision ($Collision$):** $-50.0$ points.
   * **Out of Bounds ($Over Map$):** $-50.0$ points.
   * **Timeout ($Truncated$):** $-50.0$ points.
