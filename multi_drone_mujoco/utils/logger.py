"""Logger utility for recording drone flight data.

Records state, actions, and rewards for analysis and plotting.
"""

import os
from datetime import datetime

import numpy as np


class Logger:
    """Flight data logger with CSV export and basic plotting.

    Logs per-drone: timestamps, positions, velocities, RPY, angular velocities,
    actions (RPMs), and rewards.
    """

    def __init__(self, num_drones: int, logging_freq: int = 48, output_folder: str = "results"):
        self.NUM_DRONES = num_drones
        self.LOGGING_FREQ = logging_freq
        self.OUTPUT_FOLDER = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Pre-allocate storage (will grow as needed)
        self._timestamps = []
        self._positions = [[] for _ in range(num_drones)]
        self._velocities = [[] for _ in range(num_drones)]
        self._rpys = [[] for _ in range(num_drones)]
        self._ang_vels = [[] for _ in range(num_drones)]
        self._actions = [[] for _ in range(num_drones)]
        self._rewards = []

    def log(
        self,
        drone: int,
        timestamp: float,
        state: np.ndarray,
        action: np.ndarray = None,
        reward: float = 0.0,
    ):
        """Log a single timestep for one drone.

        Parameters
        ----------
        drone : int
            Drone index.
        timestamp : float
            Simulation time.
        state : ndarray (20,)
            State vector [pos(3), quat(4), rpy(3), vel(3), angvel(3), action(4)].
        action : ndarray (4,), optional
            Applied action (RPMs).
        reward : float
            Step reward.
        """
        if drone == 0:
            self._timestamps.append(timestamp)
            self._rewards.append(reward)

        self._positions[drone].append(state[0:3].copy())
        self._velocities[drone].append(state[10:13].copy())
        self._rpys[drone].append(state[7:10].copy())
        self._ang_vels[drone].append(state[13:16].copy())
        if action is not None:
            self._actions[drone].append(action.copy())
        else:
            self._actions[drone].append(state[16:20].copy())

    def save_to_csv(self, filename: str = None):
        """Save logged data to CSV files (one per drone)."""
        if filename is None:
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for d in range(self.NUM_DRONES):
            filepath = os.path.join(self.OUTPUT_FOLDER, f"{filename}_drone{d}.csv")
            positions = np.array(self._positions[d])
            velocities = np.array(self._velocities[d])
            rpys = np.array(self._rpys[d])
            ang_vels = np.array(self._ang_vels[d])
            actions = np.array(self._actions[d])
            timestamps = np.array(self._timestamps[:len(positions)])

            header = "t,x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz,rpm0,rpm1,rpm2,rpm3"
            data = np.column_stack([
                timestamps.reshape(-1, 1),
                positions, velocities, rpys, ang_vels, actions,
            ])
            np.savetxt(filepath, data, delimiter=",", header=header, comments="")
            print(f"[Logger] Saved drone {d} data to {filepath}")

    def save_rewards(self, filename: str = None):
        """Save rewards to file."""
        if filename is None:
            filename = f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.OUTPUT_FOLDER, filename)
        np.savetxt(filepath, self._rewards, delimiter=",", header="reward", comments="")

    def plot(self, drone: int = 0, show: bool = True):
        """Plot flight data for a drone.

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Logger] matplotlib not available for plotting")
            return

        positions = np.array(self._positions[drone])
        velocities = np.array(self._velocities[drone])
        rpys = np.array(self._rpys[drone])
        actions = np.array(self._actions[drone])
        timestamps = np.array(self._timestamps[:len(positions)])

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Position
        axes[0].plot(timestamps, positions[:, 0], label="x")
        axes[0].plot(timestamps, positions[:, 1], label="y")
        axes[0].plot(timestamps, positions[:, 2], label="z")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[0].grid(True)

        # Velocity
        axes[1].plot(timestamps, velocities[:, 0], label="vx")
        axes[1].plot(timestamps, velocities[:, 1], label="vy")
        axes[1].plot(timestamps, velocities[:, 2], label="vz")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()
        axes[1].grid(True)

        # Attitude
        axes[2].plot(timestamps, np.degrees(rpys[:, 0]), label="roll")
        axes[2].plot(timestamps, np.degrees(rpys[:, 1]), label="pitch")
        axes[2].plot(timestamps, np.degrees(rpys[:, 2]), label="yaw")
        axes[2].set_ylabel("Attitude (deg)")
        axes[2].legend()
        axes[2].grid(True)

        # Actions
        if len(actions) > 0:
            for m in range(4):
                axes[3].plot(timestamps[:len(actions)], actions[:, m], label=f"motor{m}")
            axes[3].set_ylabel("RPM")
            axes[3].legend()
            axes[3].grid(True)

        axes[3].set_xlabel("Time (s)")
        plt.suptitle(f"Drone {drone} Flight Log")
        plt.tight_layout()

        savepath = os.path.join(self.OUTPUT_FOLDER, f"plot_drone{drone}.png")
        plt.savefig(savepath, dpi=150)
        print(f"[Logger] Plot saved to {savepath}")

        if show:
            plt.show()

    def plot_3d(self, show: bool = True):
        """3D trajectory plot for all drones."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            print("[Logger] matplotlib not available")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.tab10(np.linspace(0, 1, self.NUM_DRONES))
        for d in range(self.NUM_DRONES):
            positions = np.array(self._positions[d])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color=colors[d], label=f"Drone {d}")
            ax.scatter(*positions[0], color=colors[d], marker="o", s=50)
            ax.scatter(*positions[-1], color=colors[d], marker="x", s=50)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.title("3D Trajectories")

        savepath = os.path.join(self.OUTPUT_FOLDER, "plot_3d.png")
        plt.savefig(savepath, dpi=150)
        print(f"[Logger] 3D plot saved to {savepath}")
        if show:
            plt.show()
