import os
import sys
import numpy as np
import random
import math
import mujoco
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.append(path)

from multi_drone_mujoco.envs.base_aviary import BaseAviary, CF2_MESH_DIR, DRONE_PARAMS
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
import config

MAX_STEPS = 10000

def _generate_drone_nav_xml(num_drones, drone_model, init_xyzs, init_rpys, obstacle_positions, goal_position, vision=True, timestep=1/240):
    meshdir = str(CF2_MESH_DIR)
    params = DRONE_PARAMS[drone_model]

    visual_meshes = "\n".join(f'    <mesh file="{meshdir}/cf2_{i}.obj" name="cf2_vis_{i}"/>' for i in range(7))
    collision_meshes = "\n".join(f'    <mesh file="{meshdir}/cf2_collision_{i}.obj" name="cf2_col_{i}"/>' for i in range(32))

    mass = params["mass"]
    ixx, iyy, izz = params["ixx"], params["iyy"], params["izz"]
    L = params["arm_length"]

    drone_bodies = ""
    sensors = ""

    for d in range(num_drones):
        x, y, z = init_xyzs[d]
        r, p_angle, yaw = init_rpys[d]
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p_angle / 2), np.sin(p_angle / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        prefix = f"drone{d}"

        prop_offsets = [
            (L / np.sqrt(2), L / np.sqrt(2), 0),
            (-L / np.sqrt(2), L / np.sqrt(2), 0),
            (-L / np.sqrt(2), -L / np.sqrt(2), 0),
            (L / np.sqrt(2), -L / np.sqrt(2), 0),
        ]

        prop_sites = "".join(f'      <site name="{prefix}_prop{pi}" pos="{px} {py} {pz}" group="5"/>\n' for pi, (px, py, pz) in enumerate(prop_offsets))

        drone_bodies += f"""
    <body name="{prefix}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{prefix}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{prefix}_collision" type="cylinder" size="{params['collision_r']} {params['collision_h'] / 2}" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom mesh="cf2_vis_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_vis_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_vis_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_vis_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_vis_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_vis_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_vis_6" material="white" class="visual"/>
      <!-- Mũi tên con trỏ màu trắng siêu nổi bật (phóng to 10x) đi theo Drone -->
      <geom name="{prefix}_arrow_stem" type="cylinder" pos="0 0 1.8" size="0.12 0.7" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_arrow_pointer" type="sphere" pos="0 0 0.8" size="0.4" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <site name="{prefix}_center" pos="0 0 0" group="5"/>
{prop_sites}"""

        if vision:
            drone_bodies += f'      <camera name="{prefix}_cam" pos="0.02 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>\n'
        drone_bodies += "    </body>\n"

        sensors += f"""
    <gyro name="{prefix}_gyro" site="{prefix}_center"/>
    <accelerometer name="{prefix}_acc" site="{prefix}_center"/>
    <framequat name="{prefix}_quat" objtype="site" objname="{prefix}_center"/>
    <framepos name="{prefix}_pos" objtype="site" objname="{prefix}_center"/>
    <framelinvel name="{prefix}_vel" objtype="site" objname="{prefix}_center"/>
    <frameangvel name="{prefix}_angvel" objtype="site" objname="{prefix}_center"/>"""

    obstacle_bodies = ""
    for i in range(10):
        pos = obstacle_positions[i] if i < len(obstacle_positions) else [0, 0]
        obstacle_bodies += f"""
    <body name="obstacle_{i}" pos="{pos[0]} {pos[1]} 1.5">
      <geom type="cylinder" size="1.0 1.5" rgba="0 0.7 0.3 1" contype="1" conaffinity="1"/>
    </body>"""

    goal_x, goal_y = goal_position[0], goal_position[1]
    obstacle_bodies += f"""
    <body name="goal_pole" pos="{goal_x} {goal_y} 1.5">
      <geom type="cylinder" size="0.1 1.5" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    </body>"""

    xml = f"""<mujoco model="drone_dqn_nav">
  <option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="{timestep}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{visual_meshes}
{collision_meshes}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="15 15 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_bodies}{obstacle_bodies}
  </worldbody>

  <sensor>
{sensors}
  </sensor>
</mujoco>"""
    return xml

class DroneEnv(BaseAviary):
    def __init__(self,
                 gui=True,
                 initial_xyzs=None,
                 initial_rpys=np.array([[0.0, 0.0, 0.0]]),
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 physics=Physics.MJC,
                 sim_freq=240,
                 ctrl_freq=48,
                 record=False,
                 obstacles=True,
                 output_folder='results',
                 show_lidar=True):
        self.show_lidar = show_lidar
        self.obstacles_position = []
        self.collision = False
        self.win = False
        self.over_step = False
        self.over_map = False
        self.step_count = 0
        self.reward = 0
        self.start, self.goal = self.add_start_goal()
        self.obstacles_position = self._generate_safe_obstacles()

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         physics=physics,
                         sim_freq=sim_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=False,
                         obs_type=ObservationType.KIN,
                         act_type=ActionType.RPM,
                         vision_attributes=True,
                         initial_xyzs=self.start,
                         initial_rpys=initial_rpys)
        
        xml_str = _generate_drone_nav_xml(
            num_drones=num_drones,
            drone_model=drone_model,
            init_xyzs=self.INIT_XYZS,
            init_rpys=self.INIT_RPYS,
            obstacle_positions=self.obstacles_position,
            goal_position=self.goal[0],
            vision=True,
            timestep=self.SIM_TIMESTEP
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.IMG_RES = np.array([64, 64])

    def _generate_safe_obstacles(self):
        JITTERING = 1.0
        MIN_DIST_TO_GOAL = 2.5
        MIN_DIST_TO_START = 2.0
        
        safe_obstacles = []
        candidates = random.sample(config.obstacle_position, len(config.obstacle_position))
        
        for pos in candidates:
            if len(safe_obstacles) >= 10:
                break
            
            placed = False
            for _ in range(20):
                jitter_x = random.uniform(-JITTERING, JITTERING)
                jitter_y = random.uniform(-JITTERING, JITTERING)
                new_x = pos[0] + jitter_x
                new_y = pos[1] + jitter_y
                
                dist_to_start = math.hypot(new_x - self.start[0][0], new_y - self.start[0][1])
                dist_to_goal = math.hypot(new_x - self.goal[0][0], new_y - self.goal[0][1])
                
                if dist_to_start >= MIN_DIST_TO_START and dist_to_goal >= MIN_DIST_TO_GOAL:
                    safe_obstacles.append([new_x, new_y, 0])
                    placed = True
                    break
            
            if not placed:
                dist_to_start = math.hypot(pos[0] - self.start[0][0], pos[1] - self.start[0][1])
                dist_to_goal = math.hypot(pos[0] - self.goal[0][0], pos[1] - self.goal[0][1])
                if dist_to_start >= MIN_DIST_TO_START and dist_to_goal >= MIN_DIST_TO_GOAL:
                    safe_obstacles.append([pos[0], pos[1], 0])

        return safe_obstacles

    def add_start_goal(self):
        start_index = random.choice(config.start_space)
        goal_index = random.choice(config.goal_space)
        start_point = np.array([[start_index[0], start_index[1], 0.1]])
        goal_point = np.array([[goal_index[0], goal_index[1], 0.1]])
        return start_point, goal_point

    def _computeReward(self):
        self.reward = 0
        self.collision = self.check_collision()
        self.win = self.check_win()
        self.over_map = self.check_overmap()
        state = self._getDroneStateVector(0)
        
        current_dist_to_goal = np.linalg.norm(state[0:3] - self.goal[0])
        progress = self.previous_dist - current_dist_to_goal

        sensors = self.get_raycast_sensors()
        min_sensor_dist = float(np.min(sensors))
        d_min_meter = min_sensor_dist * 5.0

        front_dist_meter = float(np.min(sensors[3:6])) * 5.0

        if progress <= 0:
            reward1 = -0.005
        else:
            if front_dist_meter < 1.8:
                reward1 = -3.0 * progress
            else:
                reward1 = 5.0 * progress

        D_DANGER = 2.0
        if d_min_meter < D_DANGER:
            repulsive_penalty = 0.04 * ((D_DANGER - d_min_meter) / D_DANGER) ** 2
            reward1 -= repulsive_penalty

        if self.collision: reward1 -= 25.0
        if self.over_map:  reward1 -= 25.0
        if self.win:       reward1 += 50.0

        self.previous_dist = current_dist_to_goal
        return float(reward1)

    def _computeInfo(self):
        status1 = "Drone đã va chạm với cột. " if self.collision else "Drone chưa có va chạm. "
        status2 = "Drone đã tới đích thành công . " if self.win else "Drone chưa tới đích. "
        return status1 + status2

    def _computeTruncated(self):
        status1 = "WIN" if self.win else "NONE"
        status2 = "OVER_STEP" if self.over_step else "NONE"
        status3 = "OVER_MAP" if self.over_map else "NONE"
        check = False if ((status1 == "NONE") and (status2 == "NONE") and (status3 == "NONE")) else True
        return (check, status1, status2, status3)

    def _computeTerminated(self):
        return bool(self.collision)

    def _computeObs(self):
        return self._getDroneStateVector(0)

    def update_mujoco_bodies(self):
        for i, pos in enumerate(self.obstacles_position):
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            if b_id >= 0:
                self.model.body_pos[b_id] = [pos[0], pos[1], 1.5]
                
        g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_pole")
        if g_id >= 0:
            self.model.body_pos[g_id] = [self.goal[0][0], self.goal[0][1], 1.5]

        mujoco.mj_forward(self.model, self.data)

    def reset(self, options=None, seed=None):
        self.collision = False
        self.win = False
        self.over_step = False
        self.over_map = False
        self.step_count = 0
        self.reward = 0
        self.start, self.goal = self.add_start_goal()
        self.INIT_XYZS = self.start
        self.obstacles_position = self._generate_safe_obstacles()

        self.previous_dist = np.linalg.norm(self.goal - self.start)
        res = super().reset(seed=seed)
        self.update_mujoco_bodies()
        if self.GUI:
            self.render()
        return res

    def _preprocessAction(self, action):
        return np.array(action).flatten()

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def _actionSpace(self):
        return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action=action)
        if self.GUI:
            self.render()
        img = self._getDroneImages(nth_drone=0)
        self.step_count += 1
        return obs, img, reward, terminated, truncated, info

    def check_collision(self):
        state = self._getDroneStateVector(0)
        drone_x, drone_y, drone_z = state[0:3]
        
        if drone_z < 0.05:
            return True

        CYLINDER_RADIUS = 1.0
        DRONE_SAFETY_MARGIN = 0.1
        COLLISION_THRESHOLD = CYLINDER_RADIUS + DRONE_SAFETY_MARGIN
        
        for pos in self.obstacles_position:
            dist_xy = math.hypot(drone_x - pos[0], drone_y - pos[1])
            if dist_xy <= COLLISION_THRESHOLD:
                return True
        return False

    def check_win(self):
        state = self._getDroneStateVector(0)
        cur_x, cur_y, _ = state[0:3]
        goal_x, goal_y, _ = self.goal[0][0:3]

        if abs(goal_x - cur_x) <= 1.0 and abs(goal_y - cur_y) <= 1.0:
            return True
        return False

    def check_overmap(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 6.5 or abs(state[1]) > 12.0:
            return True
        return False

    def get_raycast_sensors(self, nth_drone=0):
        drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")
        drone_pos = self.data.xpos[drone_id]
        R = np.array(self.data.xmat[drone_id]).reshape(3, 3)
        MAX_RANGE = 5.0

        angles_rad = [
             math.pi / 2,
             3 * math.pi / 8,
             math.pi / 4,
             math.pi / 8,
              0.0,
            -math.pi / 8,
            -math.pi / 4,
            -3 * math.pi / 8,
            -math.pi / 2,
        ]

        sensor_results = []
        self._last_lidar_rays = []
        geom_id = np.zeros(1, dtype=np.int32)

        for angle in angles_rad:
            vec_body = np.array([math.sin(angle), -math.cos(angle), 0.0], dtype=np.float64)
            vec_world = R @ vec_body

            dist = mujoco.mj_ray(self.model, self.data, drone_pos, vec_world, None, 1, drone_id, geom_id)

            if dist != -1 and dist <= MAX_RANGE:
                hit_geom = geom_id[0]
                body_id = self.model.geom_bodyid[hit_geom]
                b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if b_name and b_name.startswith("obstacle_"):
                    frac = dist / MAX_RANGE
                    sensor_results.append(frac)
                    self._last_lidar_rays.append({
                        "from": drone_pos.copy(),
                        "to": drone_pos + vec_world * dist,
                        "is_hit": True,
                        "dist": dist
                    })
                    continue

            sensor_results.append(1.0)
            self._last_lidar_rays.append({
                "from": drone_pos.copy(),
                "to": drone_pos + vec_world * MAX_RANGE,
                "is_hit": False,
                "dist": MAX_RANGE
            })

        return np.array(sensor_results, dtype=np.float32)

    def _draw_lidar_rays(self):
        if self._viewer is None or not hasattr(self._viewer, "user_scn"):
            return
        
        if not hasattr(self, "_last_lidar_rays") or not self._last_lidar_rays:
            self.get_raycast_sensors()

        scn = self._viewer.user_scn
        scn.ngeom = 0

        for ray in self._last_lidar_rays:
            p_from = ray["from"]
            p_to = ray["to"]
            is_hit = ray["is_hit"]
            dist = ray["dist"]

            if scn.ngeom >= scn.maxgeom - 2:
                break

            if is_hit:
                if dist < 1.5:
                    rgba = np.array([1.0, 0.1, 0.1, 0.95], dtype=np.float32)
                else:
                    rgba = np.array([1.0, 0.7, 0.0, 0.85], dtype=np.float32)
                radius = 0.015
            else:
                rgba = np.array([0.0, 0.85, 1.0, 0.35], dtype=np.float32)
                radius = 0.008

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p_from, p_to)
            scn.ngeom += 1

            if is_hit and scn.ngeom < scn.maxgeom:
                hit_geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(hit_geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0.06, 0.06]), p_to, np.eye(3).flatten(), np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
                scn.ngeom += 1

    def toggle_lidar(self):
        """Bật/tắt hiển thị tia laser LiDAR 3D trên GUI."""
        self.show_lidar = not getattr(self, "show_lidar", True)
        if not self.show_lidar and self._viewer is not None and hasattr(self._viewer, "user_scn"):
            self._viewer.user_scn.ngeom = 0
            self._viewer.sync()
        return self.show_lidar

    def render(self, camera_mode=None, track_drone_id=0):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if getattr(self, "show_lidar", True):
                self._draw_lidar_rays()
            else:
                if hasattr(self._viewer, "user_scn"):
                    self._viewer.user_scn.ngeom = 0
            self._viewer.sync()
        else:
            return super().render(camera_mode=camera_mode, track_drone_id=track_drone_id)
