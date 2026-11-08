# Testing and Evaluation Guide for 5 Aviary Environments in MuJoCo-Drones

This folder contains the complete test scripts, 3D visualization utilities, and benchmarking controllers for 5 advanced drone aviary environments built on MuJoCo physics:

1. **VelocityAviary** (3D Linear Velocity & Yaw Rate Tracking)
2. **FlyThroughAviary** (Sequential 3D Waypoint Navigation)
3. **FormationAviary** (Multi-Drone Geometric Formation Flight)
4. **RaceAviary** (Fast Lap Circuit Gate Racing)
5. **MultiAgentAviary** (PettingZoo ParallelEnv Multi-Agent Standard)

---

## 1. Directory Structure

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos\
├── controller_utils.py
├── 01_test_velocity_aviary.py
├── 02_test_fly_through_aviary.py
├── 03_test_formation_aviary.py
├── 04_test_race_aviary.py
├── 05_test_multi_agent_aviary.py
├── run_all.py
├── README.md
└── gifs/
    ├── velocity_aviary.gif
    ├── fly_through_aviary.gif
    ├── formation_aviary.gif
    ├── race_aviary.gif
    └── multi_agent_aviary.gif
```

---

## 2. Environment Specifications

| Environment | Task Type | Drone Count | Obs Dimension | Action Dimension | Primary Objective |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`VelocityAviary`** | Single-Agent RL | 1 | 16 | 4 | Track target velocity vector `[vx, vy, vz, yaw_rate]` |
| **`FlyThroughAviary`** | Single / Multi | 1 (or N) | 18 | 4 | Sequentially traverse 3D spatial waypoints |
| **`FormationAviary`** | Cooperative Multi-Drone | 3 (or N) | 54 (18 × 3) | 12 (4 × 3) | Maintain triangular formation while tracking moving center |
| **`RaceAviary`** | Agile Racing RL | 1 (or N) | 21 | 4 | Race through circuit gates as fast as possible |
| **`MultiAgentAviary`** | PettingZoo ParallelEnv | N (1 agent/drone) | 13 (per agent) | 4 (per agent) | Decentralized MARL altitude holding and stabilization |

---

### Technical Details

#### 1. VelocityAviary
- **Observation Space (16-dim)**:
  - `pos` (3): Spatial position [x, y, z]
  - `rpy` (3): Euler angles roll, pitch, yaw
  - `vel` (3): Linear velocity [vx, vy, vz]
  - `ang_v` (3): Angular velocity [wx, wy, wz]
  - `TARGET_VEL` (4): Desired target velocity [vx_des, vy_des, vz_des, yaw_rate_des]
- **Action Space (4-dim)**: Normalized motor RPMs in range `[-1, 1]`.
- **Reward Function**: Penalizes linear velocity error `||vel - TARGET_VEL[:3]||` and yaw rate error; bonus when velocity error < 0.05 m/s; penalty of -100 on collision or tilt exceeding 90 degrees.

#### 2. FlyThroughAviary
- **Observation Space (18-dim)**: Kinematic state (12) + Next waypoint position (3) + Relative vector to waypoint `rel_wp = wp - pos` (3).
- **Trajectory**: Sequential series of 3D waypoints with visual markers.
- **Waypoint Detection**: When distance to target waypoint < `WAYPOINT_RADIUS` (0.15m), targets the next waypoint and awards **+10.0 points**.

#### 3. FormationAviary
- **Observation Space (18 × N dim)**: Full states of N drones and respective target formation offsets.
- **Offset Geometry**: Each drone preserves relative offset from geometric centroid (equilateral triangle radius 0.3m).
- **Reward Function**: Jointly evaluates centroid trajectory tracking and penalizes deviation from target inter-agent distances.

#### 4. RaceAviary
- **Observation Space (21-dim)**: Kinematic state (12) + Next gate center (3) + Relative gate vector (3) + Subsequent gate center (3) for previewing upcoming turns.
- **Reward Function**: **+20.0 points** for crossing each gate portal + velocity bonus `2.0 * ||vel||` encouraging high speed through gates.

#### 5. MultiAgentAviary
- **Architecture**: PettingZoo `ParallelEnv` interface (`env.possible_agents = ['drone0', 'drone1', 'drone2']`).
- **Dict I/O**:
  - `actions = {"drone0": act0, "drone1": act1, "drone2": act2}`
  - `observations, rewards, terminations, truncations, infos = env.step(actions)`
- **Compatibility**: Directly compatible with multi-agent RL libraries such as Tianshou, CleanRL, Ray RLlib, or Stable-Baselines3 via SuperSuit.

---

## 3. Running Demonstrations

Navigate to the test directory:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos
```

### Interactive 3D Viewer (`--gui`)

Controls in MuJoCo viewer:
- **Left click + drag**: Rotate camera
- **Right click + drag**: Zoom in / out
- **Middle click / scroll**: Pan camera
- **Space key**: Pause / resume simulation

```powershell
python 01_test_velocity_aviary.py --gui
python 02_test_fly_through_aviary.py --gui
python 03_test_formation_aviary.py --gui
python 04_test_race_aviary.py --gui
python 05_test_multi_agent_aviary.py --gui
```

---

### Record Animated GIFs (`--record`)

Renders frames offscreen and saves `.gif` files into `gifs/`:

```powershell
python 01_test_velocity_aviary.py --record
python 02_test_fly_through_aviary.py --record
python 03_test_formation_aviary.py --record
python 04_test_race_aviary.py --record
python 05_test_multi_agent_aviary.py --record
```

---

### Run All Environments via `run_all.py`

```powershell
python run_all.py
python run_all.py --record
python run_all.py --env fly_through --gui
```

---

### Random Action Baseline (`--random`)

By default, scripts execute autonomous cascaded PID tracking for stable demonstration. Use `--random` to evaluate unconditioned policy exploration:

```powershell
python 01_test_velocity_aviary.py --random
```
