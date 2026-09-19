# MJ-drones-gym

**MuJoCo-based multi-drone Gymnasium environments for single and multi-agent reinforcement learning of quadcopter control.**

High-fidelity quadcopter simulation with GPU-vectorized environments, Dryden wind turbulence, domain randomization, obstacle generation, and curriculum learning — all built on [MuJoCo](https://mujoco.org/).

## Features

- **MuJoCo physics** — faster and more accurate than PyBullet
- **Gymnasium API** — drop-in compatible with stable-baselines3, CleanRL, etc.
- **Multi-drone support** — N arbitrary drones with inter-drone effects
- **Aerodynamic effects** — ground effect, drag, downwash (individually toggleable)
- **Multiple action types** — RPM, normalized thrust, velocity, PID waypoint
- **Multiple observation types** — kinematics (state vector), RGB camera
- **PID controllers** — tuned cascaded position/attitude PID (PIDControl + DSLPIDControl)
- **PettingZoo multi-agent** — parallel environment wrapper for MARL
- **7 task environments** — hover, velocity tracking, waypoint navigation, formation, racing, and more
- **SB3 examples** — ready-to-run PPO training scripts

## Installation

```bash
git clone https://github.com/nghuucam/MuJoCo-Drones.git
cd MuJoCo-Drones/
pip install -e .          # core
pip install -e ".[all]"   # with RL, MARL, and visualization extras
```

### Requirements
- Python ≥ 3.8
- MuJoCo ≥ 3.0
- Gymnasium ≥ 0.29
- NumPy ≥ 1.21

## Quick Start

### PID Control

```python
import numpy as np
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.utils.enums import Physics

env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
ctrl = PIDControl(env)
env.reset()

target = np.array([0.5, 0.3, 1.0])
for _ in range(4800):
    rpm, _, _ = ctrl.computeControl(
        env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
        env.vel[0], env.ang_v[0], target
    )
    env.step(rpm.flatten())

print(f"Final position error: {np.linalg.norm(env.pos[0] - target):.4f} m")
env.close()
```

### Reinforcement Learning (SB3 PPO)

```python
from stable_baselines3 import PPO
from multi_drone_mujoco.envs.hover_aviary import HoverAviary

env = HoverAviary(ctrl_freq=48)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("hover_ppo")
```

### Multi-Agent RL (PettingZoo)

```python
from multi_drone_mujoco.envs.multi_agent_aviary import MultiAgentAviary

env = MultiAgentAviary(num_drones=3)
env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terms, truncs, infos = env.step(actions)
```

## Examples

```bash
cd multi_drone_mujoco/examples/
python pid.py          # PID hover + velocity tracking + multi-drone
python downwash.py     # downwash effect demonstration
python learn.py        # SB3 PPO training (single + multi hover)
python play.py         # visualize trained policy
```

## Environments

| Environment | Obs Dim | Action | Description |
|---|---|---|---|
| `HoverAviary` | 12 | 4 (normalized RPM) | Hover at z=1.0 |
| `VelocityAviary` | 16 | 4 (normalized RPM) | Track velocity commands |
| `MultiHoverAviary` | 13×N | 4×N | N drones at different heights |
| `FlyThroughAviary` | 18 | 4 | Navigate through waypoints |
| `FormationAviary` | 18×N | 4×N | Formation flying along a path |
| `RaceAviary` | 21 | 4 | Gate racing with lap timing |
| `MultiAgentAviary` | per-agent | per-agent | PettingZoo parallel wrapper |

## Physics Modes

| Mode | Description |
|---|---|
| `Physics.MJC` | Pure MuJoCo (force injection via xfrc_applied) |
| `Physics.DYN` | Explicit dynamics (Euler integration) |
| `Physics.MJC_GND` | MuJoCo + ground effect |
| `Physics.MJC_DRAG` | MuJoCo + aerodynamic drag |
| `Physics.MJC_DW` | MuJoCo + downwash |
| `Physics.MJC_GND_DRAG_DW` | MuJoCo + all aerodynamic effects |

## Tests

```bash
pytest multi_drone_mujoco/tests/ -v
```

## Project Structure

```
multi_drone_mujoco/
├── envs/
│   ├── base_aviary.py          # Core physics engine + Gymnasium env
│   ├── hover_aviary.py         # Single-drone hover task
│   ├── velocity_aviary.py      # Velocity tracking task
│   ├── multi_hover_aviary.py   # Multi-drone hover
│   ├── fly_through_aviary.py   # Waypoint navigation
│   ├── formation_aviary.py     # Formation flying
│   ├── race_aviary.py          # Gate racing
│   └── multi_agent_aviary.py   # PettingZoo wrapper
├── control/
│   ├── pid_control.py          # Cascaded PID controller
│   └── dsl_pid_control.py      # Enhanced PID with anti-windup
├── utils/
│   ├── enums.py                # DroneModel, Physics, ActionType, etc.
│   └── logger.py               # CSV logging + matplotlib plotting
├── examples/
│   ├── pid.py                  # PID control demos
│   ├── downwash.py             # Downwash visualization
│   ├── learn.py                # SB3 PPO training
│   └── play.py                 # Trained model playback
├── tests/
│   ├── test_envs.py            # Environment tests
│   ├── test_control.py         # Controller tests
│   └── test_multi_agent.py     # MARL tests
└── setup.py
```

## Differences from gym-pybullet-drones

| | gym-pybullet-drones | gym-mujoco-drones |
|---|---|---|
| Physics | PyBullet | MuJoCo (faster, more accurate) |
| Rendering | PyBullet GUI | MuJoCo viewer / offscreen RGB |
| Firmware SITL | Betaflight, CF firmware | — |
| Task environments | 2 (Hover, MultiHover) | 7 (+ velocity, waypoint, formation, race) |
| Multi-agent | Custom | PettingZoo standard |

## Citation

If you use this work, please cite:

```bibtex
@misc{tayal2026mujocodronesgym,
  title={MuJoCo-Drones-Gym: A GPU-Accelerated Multi-Drone Simulator for Control and Reinforcement Learning}, 
      author={Manan Tayal},
      year={2026},
      eprint={2606.08039},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.08039}, 
}
```

## Acknowledgements

- [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) — inspiration for the environment API and task design
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Bitcraze Crazyflie 2.x MJCF model
- [Bitcraze](https://www.bitcraze.io/) — Crazyflie 2.x hardware platform and firmware parameters

## License

MIT
