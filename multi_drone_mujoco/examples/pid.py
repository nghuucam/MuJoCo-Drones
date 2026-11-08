"""Example: PID-controlled hover and position tracking.

Demonstrates the PID controller flying a single drone to a target position.
"""

import numpy as np

from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
from multi_drone_mujoco.utils.logger import Logger


def pid_hover():
    """Hover a single drone at z=1.0 using PID control."""
    print("=" * 60)
    print("PID Hover Example")
    print("=" * 60)

    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
    )

    ctrl = PIDControl(env)
    logger = Logger(num_drones=1, logging_freq=48, output_folder="results/pid_hover")

    obs, info = env.reset()
    target_pos = np.array([0.0, 0.0, 1.0])

    for step in range(480):  # 10 seconds at 48Hz
        state = env._getDroneStateVector(0)
        rpm, pos_e, yaw_e = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=env.pos[0],
            cur_quat=env.quat[0],
            cur_vel=env.vel[0],
            cur_ang_vel=env.ang_v[0],
            target_pos=target_pos,
        )

        obs, reward, terminated, truncated, info = env.step(rpm.flatten())
        logger.log(0, step * env.CTRL_TIMESTEP, env._getDroneStateVector(0), rpm)

        if step % 48 == 0:
            print(f"  t={step * env.CTRL_TIMESTEP:5.1f}s | "
                  f"pos=[{env.pos[0,0]:+.3f}, {env.pos[0,1]:+.3f}, {env.pos[0,2]:+.3f}] | "
                  f"err={np.linalg.norm(pos_e):.4f}")

        if terminated:
            print("  TERMINATED (crash)")
            break

    logger.save_to_csv("pid_hover")
    env.close()
    print(f"  Final position: {env.pos[0]}")
    print(f"  Height error: {abs(env.pos[0,2] - 1.0):.4f} m")
    print()


def pid_velocity():
    """Track velocity commands using DSL PID controller."""
    print("=" * 60)
    print("PID Velocity Tracking Example")
    print("=" * 60)

    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        initial_xyzs=np.array([[0, 0, 0.5]]),
    )

    ctrl = DSLPIDControl(env)
    obs, info = env.reset()

    # Fly a square
    waypoints = [
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    wp_idx = 0

    for step in range(960):  # 20 seconds
        target = waypoints[wp_idx]
        dist = np.linalg.norm(env.pos[0] - target)
        if dist < 0.1 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            print(f"  Reached waypoint {wp_idx}!")

        rpm, pos_e, _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=env.pos[0],
            cur_quat=env.quat[0],
            cur_vel=env.vel[0],
            cur_ang_vel=env.ang_v[0],
            target_pos=target,
        )

        obs, _, terminated, _, _ = env.step(rpm.flatten())
        if terminated:
            print("  CRASHED")
            break

    env.close()
    print(f"  Waypoints reached: {wp_idx + 1}/{len(waypoints)}")
    print()


def multi_drone_pid():
    """Multiple drones hovering at different heights."""
    print("=" * 60)
    print("Multi-Drone PID Example (3 drones)")
    print("=" * 60)

    num_drones = 3
    targets = [
        np.array([0.0, 0.0, 0.8]),
        np.array([0.3, 0.0, 1.0]),
        np.array([0.6, 0.0, 1.2]),
    ]

    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=num_drones,
        physics=Physics.MJC_GND_DRAG_DW,  # Full physics!
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        initial_xyzs=np.array([[0, 0, 0.1], [0.3, 0, 0.1], [0.6, 0, 0.1]]),
    )

    controllers = [PIDControl(env) for _ in range(num_drones)]
    logger = Logger(num_drones=num_drones, output_folder="results/multi_pid")
    obs, _ = env.reset()

    for step in range(480):
        rpms = np.zeros((num_drones, 4))
        for i in range(num_drones):
            rpms[i], _, _ = controllers[i].computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=env.pos[i],
                cur_quat=env.quat[i],
                cur_vel=env.vel[i],
                cur_ang_vel=env.ang_v[i],
                target_pos=targets[i],
            )
            logger.log(i, step * env.CTRL_TIMESTEP, env._getDroneStateVector(i), rpms[i])

        obs, _, terminated, _, _ = env.step(rpms.flatten())

        if step % 96 == 0:
            errors = [np.linalg.norm(env.pos[i] - targets[i]) for i in range(num_drones)]
            print(f"  t={step * env.CTRL_TIMESTEP:5.1f}s | errors: {[f'{e:.3f}' for e in errors]}")

        if terminated:
            break

    logger.save_to_csv("multi_pid")
    env.close()
    print(f"  Final errors: {[f'{np.linalg.norm(env.pos[i] - targets[i]):.4f}' for i in range(num_drones)]}")
    print()


if __name__ == "__main__":
    pid_hover()
    pid_velocity()
    multi_drone_pid()
    print("All PID examples completed!")
