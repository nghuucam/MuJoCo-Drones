"""Generate demo GIFs for MJ-drones-gym features."""
import os
import numpy as np
from PIL import Image

os.makedirs("/root/multi_drone_mujoco/demo_gifs", exist_ok=True)


def save_gif(frames, path, fps=20):
    """Save list of numpy arrays as a GIF."""
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0)
    print(f"  Saved: {path} ({len(frames)} frames)")


def warmup(env, ctrl, target, steps=200, num_drones=1):
    """Run PID for `steps` to let drones stabilize before recording."""
    for _ in range(steps):
        if num_drones == 1:
            rpm, _, _ = ctrl.computeControl(
                env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
                env.vel[0], env.ang_v[0], target
            )
            env.step(rpm.flatten())
        else:
            all_rpm = []
            for d in range(num_drones):
                t = target[d] if target.ndim > 1 else target
                rpm, _, _ = ctrl[d].computeControl(
                    env.CTRL_TIMESTEP, env.pos[d], env.quat[d],
                    env.vel[d], env.ang_v[d], t
                )
                all_rpm.append(rpm.flatten())
            env.step(np.vstack(all_rpm))


def demo_hover():
    """Single drone hover with PID — all 3 camera modes."""
    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.control.pid_control import PIDControl

    print("Rendering: Hover (3 camera modes)...")
    env = HoverAviary(ctrl_freq=48, render_mode="rgb_array", target_height=1.0)
    ctrl = PIDControl(env)
    env.reset()

    target = np.array([0.0, 0.0, 1.0])

    # Warmup: let drone stabilize at hover
    warmup(env, ctrl, target, steps=300)

    # Record with tracking camera
    frames_track = []
    for i in range(120):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
            env.vel[0], env.ang_v[0], target
        )
        env.step(rpm.flatten())
        if i % 2 == 0:
            frames_track.append(env.render(camera_mode="track"))

    save_gif(frames_track, "/root/multi_drone_mujoco/demo_gifs/hover_track.gif")

    # Record with fixed camera
    env.reset()
    warmup(env, ctrl, target, steps=300)
    frames_fixed = []
    for i in range(120):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
            env.vel[0], env.ang_v[0], target
        )
        env.step(rpm.flatten())
        if i % 2 == 0:
            frames_fixed.append(env.render(camera_mode="fixed"))

    save_gif(frames_fixed, "/root/multi_drone_mujoco/demo_gifs/hover_fixed.gif")

    # FPV (need vision camera)
    env.close()
    env = HoverAviary(ctrl_freq=48, render_mode="rgb_array", target_height=1.0)
    # Re-create with vision to have onboard camera
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.utils.enums import Physics, ObservationType, ActionType
    env_fpv = BaseAviary(
        num_drones=1, ctrl_freq=48, sim_freq=240,
        physics=Physics.MJC, render_mode="rgb_array",
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        obs_type=ObservationType.RGB,
    )
    ctrl_fpv = PIDControl(env_fpv)
    env_fpv.reset()
    warmup(env_fpv, ctrl_fpv, target, steps=300)

    frames_fpv = []
    for i in range(120):
        rpm, _, _ = ctrl_fpv.computeControl(
            env_fpv.CTRL_TIMESTEP, env_fpv.pos[0], env_fpv.quat[0],
            env_fpv.vel[0], env_fpv.ang_v[0], target
        )
        env_fpv.step(rpm.flatten())
        if i % 2 == 0:
            frames_fpv.append(env_fpv.render(camera_mode="fpv"))

    save_gif(frames_fpv, "/root/multi_drone_mujoco/demo_gifs/hover_fpv.gif")
    env_fpv.close()


def demo_multi_drone():
    """Multiple drones hovering at different heights."""
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.utils.enums import Physics

    print("Rendering: Multi-drone...")
    initial_xyzs = np.array([
        [-0.4, -0.4, 0.1],
        [0.4, -0.4, 0.1],
        [0.0, 0.4, 0.1],
    ])
    env = BaseAviary(num_drones=3, ctrl_freq=48, sim_freq=240,
                     physics=Physics.MJC, initial_xyzs=initial_xyzs,
                     render_mode="rgb_array")
    ctrls = [PIDControl(env) for _ in range(3)]
    env.reset()

    targets = np.array([
        [-0.4, -0.4, 0.8],
        [0.4, -0.4, 1.0],
        [0.0, 0.4, 1.2],
    ])

    warmup(env, ctrls, targets, steps=300, num_drones=3)

    frames = []
    for i in range(180):
        all_rpm = []
        for d in range(3):
            rpm, _, _ = ctrls[d].computeControl(
                env.CTRL_TIMESTEP, env.pos[d], env.quat[d],
                env.vel[d], env.ang_v[d], targets[d]
            )
            all_rpm.append(rpm.flatten())
        env.step(np.vstack(all_rpm))
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))

    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/multi_drone.gif")


def demo_formation():
    """Formation flying."""
    from multi_drone_mujoco.envs.formation_aviary import FormationAviary
    from multi_drone_mujoco.control.pid_control import PIDControl

    print("Rendering: Formation...")
    env = FormationAviary(num_drones=4, ctrl_freq=48, render_mode="rgb_array")
    ctrls = [PIDControl(env) for _ in range(4)]
    env.reset()

    # First stabilize at initial formation
    init_targets = np.array([
        [0.0, 0.0, 1.0]
    ] * 4) + env.FORMATION_OFFSETS
    warmup(env, ctrls, init_targets, steps=300, num_drones=4)

    # Now fly formation in a circle and record
    frames = []
    for i in range(360):
        t = (i + 300) / 48.0  # offset time for smooth motion
        cx = 0.5 * np.cos(0.3 * t)
        cy = 0.5 * np.sin(0.3 * t)
        cz = 1.0

        all_rpm = []
        for d in range(4):
            target = np.array([cx, cy, cz]) + env.FORMATION_OFFSETS[d]
            rpm, _, _ = ctrls[d].computeControl(
                env.CTRL_TIMESTEP, env.pos[d], env.quat[d],
                env.vel[d], env.ang_v[d], target
            )
            all_rpm.append(rpm.flatten())
        env.step(np.vstack(all_rpm))
        if i % 4 == 0:
            frames.append(env.render(camera_mode="track"))

    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/formation.gif")


def demo_wind():
    """Drone fighting wind turbulence."""
    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel

    print("Rendering: Wind disturbance...")
    env = HoverAviary(ctrl_freq=48, render_mode="rgb_array", target_height=1.0)
    ctrl = PIDControl(env)
    env.reset()

    target = np.array([0.0, 0.0, 1.0])

    # Stabilize first WITHOUT wind
    warmup(env, ctrl, target, steps=300)

    # Now enable wind and record the drone fighting it
    wind_cfg = WindConfig(
        model=WindModel.COMBINED,
        constant_wind=np.array([0.015, 0.008, 0.0]),
        gust_intensity=0.03,
        turbulence_intensity=3.0,
    )
    env.set_wind(wind_cfg)

    frames = []
    for i in range(240):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
            env.vel[0], env.ang_v[0], target
        )
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))

    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/wind.gif")


def demo_obstacles():
    """Drone navigating through a cluttered environment."""
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.utils.enums import Physics
    from multi_drone_mujoco.wrappers.obstacles import generate_obstacles, ObstacleConfig, ObstacleType

    print("Rendering: Obstacles (cluttered)...")
    obs_cfg = ObstacleConfig(
        obstacle_type=ObstacleType.FOREST,
        num_obstacles=30,
        arena_size=(3.0, 3.0, 2.0),
        seed=42,
    )
    obstacles = generate_obstacles(obs_cfg)

    # Build custom XML with many obstacles injected
    env = BaseAviary(
        num_drones=1, ctrl_freq=48, sim_freq=240,
        physics=Physics.MJC, render_mode="rgb_array",
        initial_xyzs=np.array([[-1.0, -1.0, 0.1]]),
        obstacles=True,
    )

    # Manually inject more obstacles into the model
    import mujoco
    for i, obs in enumerate(obstacles[:25]):
        try:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        except:
            pass  # Body doesn't exist, skip

    ctrl = PIDControl(env)
    env.reset()

    target = np.array([-1.0, -1.0, 1.0])
    warmup(env, ctrl, target, steps=200)

    # Navigate through waypoints
    waypoints = [
        np.array([-1.0, -1.0, 1.0]),
        np.array([0.0, -0.5, 1.0]),
        np.array([0.5, 0.0, 1.2]),
        np.array([0.0, 0.5, 1.0]),
        np.array([-0.5, 0.5, 0.8]),
    ]

    frames = []
    wp_idx = 0
    for i in range(360):
        target = waypoints[wp_idx]
        if np.linalg.norm(env.pos[0] - target) < 0.15 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]

        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
            env.vel[0], env.ang_v[0], target
        )
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="fixed"))

    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/obstacles.gif")


def demo_obstacles_cluttered():
    """Create env with many obstacles via XML injection."""
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.utils.enums import Physics

    print("Rendering: Cluttered obstacles (custom XML)...")

    # We'll patch the XML generation to add more obstacles
    import multi_drone_mujoco.envs.base_aviary as ba
    original_gen = ba._generate_aviary_xml

    def patched_gen(*args, **kwargs):
        xml = original_gen(*args, **kwargs)
        # Inject many obstacles before </worldbody>
        np.random.seed(42)
        obs_xml = ""
        colors = [
            "0.8 0.2 0.2 0.9",  # red
            "0.2 0.6 0.8 0.9",  # blue
            "0.2 0.8 0.3 0.9",  # green
            "0.8 0.6 0.1 0.9",  # orange
            "0.6 0.2 0.8 0.9",  # purple
        ]
        for i in range(25):
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            # Keep away from start position
            if abs(x + 1.0) < 0.3 and abs(y + 1.0) < 0.3:
                continue
            z_h = np.random.uniform(0.3, 1.2)
            r = np.random.uniform(0.03, 0.08)
            color = colors[i % len(colors)]
            obs_xml += f'    <body name="obs_{i}" pos="{x:.2f} {y:.2f} {z_h:.2f}">\n'
            obs_xml += f'      <geom type="cylinder" size="{r:.3f} {z_h:.2f}" rgba="{color}" contype="1" conaffinity="1"/>\n'
            obs_xml += f'    </body>\n'
        xml = xml.replace("  </worldbody>", obs_xml + "  </worldbody>")
        return xml

    ba._generate_aviary_xml = patched_gen

    env = BaseAviary(
        num_drones=1, ctrl_freq=48, sim_freq=240,
        physics=Physics.MJC, render_mode="rgb_array",
        initial_xyzs=np.array([[-1.0, -1.0, 0.1]]),
    )
    ctrl = PIDControl(env)
    env.reset()

    # Restore original
    ba._generate_aviary_xml = original_gen

    target = np.array([-1.0, -1.0, 1.0])
    warmup(env, ctrl, target, steps=250)

    waypoints = [
        np.array([-1.0, -1.0, 1.0]),
        np.array([-0.3, -0.3, 1.0]),
        np.array([0.5, 0.0, 1.2]),
        np.array([0.5, 1.0, 1.0]),
        np.array([1.0, 1.0, 0.8]),
    ]

    frames = []
    wp_idx = 0
    for i in range(400):
        target = waypoints[wp_idx]
        if np.linalg.norm(env.pos[0] - target) < 0.2 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]

        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
            env.vel[0], env.ang_v[0], target
        )
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="fixed"))

    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/obstacles_cluttered.gif")


def demo_domain_randomization():
    """Show domain randomization — different dynamics per episode."""
    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.wrappers import DomainRandomizationWrapper, DomainRandomizationConfig

    print("Rendering: Domain Randomization...")
    base_env = HoverAviary(ctrl_freq=48, render_mode="rgb_array", target_height=1.0)
    dr_cfg = DomainRandomizationConfig(
        mass_range=(0.6, 1.4),
        inertia_range=(0.6, 1.4),
        kf_range=(0.7, 1.3),
        km_range=(0.7, 1.3),
        action_delay_steps=2,
        motor_time_constant=0.015,
    )
    env = DomainRandomizationWrapper(base_env, dr_cfg)
    ctrl = PIDControl(base_env)

    target = np.array([0.0, 0.0, 1.0])
    all_frames = []

    # Show 3 episodes with different randomization
    for ep in range(3):
        env.reset(seed=ep * 7)
        # Warmup each episode
        for _ in range(200):
            rpm, _, _ = ctrl.computeControl(
                base_env.CTRL_TIMESTEP, base_env.pos[0], base_env.quat[0],
                base_env.vel[0], base_env.ang_v[0], target
            )
            env.step(rpm.flatten())

        # Record
        for i in range(80):
            rpm, _, _ = ctrl.computeControl(
                base_env.CTRL_TIMESTEP, base_env.pos[0], base_env.quat[0],
                base_env.vel[0], base_env.ang_v[0], target
            )
            env.step(rpm.flatten())
            if i % 2 == 0:
                all_frames.append(base_env.render(camera_mode="track"))

    env.close()
    save_gif(all_frames, "/root/multi_drone_mujoco/demo_gifs/domain_randomization.gif")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating MJ-drones-gym demo GIFs")
    print("=" * 50)

    demos = [
        ("hover", demo_hover),
        ("multi_drone", demo_multi_drone),
        ("formation", demo_formation),
        ("wind", demo_wind),
        ("obstacles_cluttered", demo_obstacles_cluttered),
        ("domain_randomization", demo_domain_randomization),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  SKIP {name}: {e}")
            traceback.print_exc()

    print("\nDone! GIFs saved to /root/multi_drone_mujoco/demo_gifs/")
