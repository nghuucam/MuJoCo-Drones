"""Example: Demonstrate downwash effect between stacked drones."""

import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType


def downwash_demo():
    """Two drones stacked vertically — bottom one experiences downwash."""
    print("=" * 60)
    print("Downwash Effect Demo")
    print("  Two drones: one hovering at z=1.5, one at z=0.7 directly below")
    print("  Physics: MJC_GND_DRAG_DW (full aerodynamic effects)")
    print("=" * 60)

    initial_xyzs = np.array([
        [0.0, 0.0, 0.7],   # Bottom drone (will experience downwash)
        [0.0, 0.0, 1.5],   # Top drone
    ])
    targets = [
        np.array([0.0, 0.0, 0.7]),
        np.array([0.0, 0.0, 1.5]),
    ]

    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        physics=Physics.MJC_GND_DRAG_DW,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        initial_xyzs=initial_xyzs,
    )

    controllers = [PIDControl(env) for _ in range(2)]
    obs, _ = env.reset()

    print("\n  Without downwash, both drones would hover perfectly.")
    print("  With downwash, bottom drone is pushed down.\n")

    for step in range(480):
        rpms = np.zeros((2, 4))
        for i in range(2):
            rpms[i], _, _ = controllers[i].computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=env.pos[i],
                cur_quat=env.quat[i],
                cur_vel=env.vel[i],
                cur_ang_vel=env.ang_v[i],
                target_pos=targets[i],
            )
        obs, _, terminated, _, _ = env.step(rpms.flatten())

        if step % 96 == 0:
            print(f"  t={step * env.CTRL_TIMESTEP:5.1f}s | "
                  f"Bottom z={env.pos[0, 2]:.4f} (target=0.7) | "
                  f"Top z={env.pos[1, 2]:.4f} (target=1.5)")

        if terminated:
            break

    bottom_err = abs(env.pos[0, 2] - 0.7)
    top_err = abs(env.pos[1, 2] - 1.5)
    print(f"\n  Bottom drone height error: {bottom_err:.4f} m (affected by downwash)")
    print(f"  Top drone height error: {top_err:.4f} m (unaffected)")
    env.close()


if __name__ == "__main__":
    downwash_demo()
