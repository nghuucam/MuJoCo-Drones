import os
import sys
import numpy as np
import random
import math
import time
import mujoco
import gymnasium as gym
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from multi_drone_mujoco.envs.base_aviary import BaseAviary, CF2_MESH_DIR, DRONE_PARAMS
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
import config

def _generate_drone_nav_xml(num_drones, drone_model, init_xyzs, init_rpys, obstacle_data, goal_position, vision=True, timestep=1/240):
    """Sinh file cấu trúc XML cho MuJoCo với camera FPV trên drone và 10 cột vật cản ngẫu nhiên độ cao."""
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
      <!-- Con trỏ màu trắng nổi bật phía trên Drone -->
      <geom name="{prefix}_arrow_stem" type="cylinder" pos="0 0 1.8" size="0.12 0.7" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_arrow_pointer" type="sphere" pos="0 0 0.8" size="0.4" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <site name="{prefix}_center" pos="0 0 0" group="5"/>
{prop_sites}"""

        if vision:
            # Camera FPV gắn trên mũi drone nhìn về phía trước (+X body)
            drone_bodies += f'      <camera name="{prefix}_cam" pos="0.03 0 0.01" xyaxes="0 -1 0 0 0 1" fovy="75"/>\n'
        drone_bodies += "    </body>\n"

        sensors += f"""
    <gyro name="{prefix}_gyro" site="{prefix}_center"/>
    <accelerometer name="{prefix}_acc" site="{prefix}_center"/>
    <framequat name="{prefix}_quat" objtype="site" objname="{prefix}_center"/>
    <framepos name="{prefix}_pos" objtype="site" objname="{prefix}_center"/>
    <framelinvel name="{prefix}_vel" objtype="site" objname="{prefix}_center"/>
    <frameangvel name="{prefix}_angvel" objtype="site" objname="{prefix}_center"/>"""

    # 10 cột vật cản với độ cao ban đầu
    obstacle_bodies = ""
    for i in range(10):
        pos = obstacle_data[i] if i < len(obstacle_data) else [0, 0, 1.5]
        h = pos[2] / 2.0  # half-height
        obstacle_bodies += f"""
    <body name="obstacle_{i}" pos="{pos[0]} {pos[1]} {h}">
      <geom name="geom_obstacle_{i}" type="cylinder" size="{config.CYLINDER_RADIUS} {h}" rgba="0 0.7 0.3 1" contype="1" conaffinity="1"/>
    </body>"""

    # Cột mốc đích màu đỏ
    goal_x, goal_y = goal_position[0], goal_position[1]
    obstacle_bodies += f"""
    <body name="goal_pole" pos="{goal_x} {goal_y} 1.5">
      <geom type="cylinder" size="0.15 1.5" rgba="1 0.1 0.1 1" contype="0" conaffinity="0"/>
    </body>"""

    xml = f"""<mujoco model="drone_ppo_nav">
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
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
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
    <light pos="0 0 4" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="15 15 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_bodies}{obstacle_bodies}
  </worldbody>

  <sensor>
{sensors}
  </sensor>
</mujoco>"""
    return xml


class DronePPOEnv(BaseAviary):
    """Môi trường Gymnasium huấn luyện Drone bằng PPO (Stable-Baselines3).
    
    Đặc điểm:
    - Kế thừa từ BaseAviary
    - Observation: Ảnh FPV RGB kích thước (64, 64, 3) dạng uint8
    - Action Space: Vector liên tục (alpha, beta, d) với alpha, beta in [0, 180] độ, d in [0, 2.0] mét
    - Điều khiển phân tầng (Hierarchical Control): PPO quyết định điểm đến, DSLPIDControl chạy vòng lặp bám theo
    - Chặn cứng chống đâm xuyên đất: z >= 0.4m
    - Chiều cao các cột vật cản ngẫu nhiên in [0.1m, 3.0m]
    """

    def __init__(self,
                 gui: bool = False,
                 record: bool = False,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.MJC,
                 sim_freq: int = 240,
                 ctrl_freq: int = 48):
        
        self.GUI = gui
        self.drone_model = drone_model
        self.step_count = 0
        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0
        
        self.collision = False
        self.win = False
        self.over_map = False
        
        # Điểm xuất phát và đích
        self.start, self.goal = self._sample_start_goal()
        # Sinh tọa độ 10 cột kèm chiều cao ngẫu nhiên
        self.obstacle_data = self._generate_safe_obstacles()
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=False,
            obs_type=ObservationType.RGB,
            act_type=ActionType.RPM,
            vision_attributes=True,
            initial_xyzs=self.start,
            initial_rpys=np.array([[0.0, 0.0, -np.pi / 2]])
        )

        # Biên dịch mô hình MuJoCo tùy chỉnh
        xml_str = _generate_drone_nav_xml(
            num_drones=1,
            drone_model=drone_model,
            init_xyzs=self.INIT_XYZS,
            init_rpys=self.INIT_RPYS,
            obstacle_data=self.obstacle_data,
            goal_position=self.goal[0],
            vision=True,
            timestep=self.SIM_TIMESTEP
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.IMG_RES = np.array([config.IMG_WIDTH, config.IMG_HEIGHT])
        
        # Bộ điều khiển PID tầng thấp
        self.control = DSLPIDControl(env=self)
        
        # Mục tiêu bám hiện tại
        self.target_pos = self.start[0].copy()
        self.target_rpy = np.array([0.0, 0.0, -np.pi / 2])
        self.previous_dist = np.linalg.norm(self.start[0] - self.goal[0])

    def _sample_start_goal(self):
        start_idx = random.choice(config.start_space)
        goal_idx = random.choice(config.goal_space)
        start_pt = np.array([[float(start_idx[0]), float(start_idx[1]), 0.1]])
        goal_pt = np.array([[float(goal_idx[0]), float(goal_idx[1]), 1.0]])
        return start_pt, goal_pt

    def _generate_safe_obstacles(self):
        """Sinh 10 vị trí vật cản an toàn và ngẫu nhiên chiều cao trong khoảng [0.1, 3.0]m."""
        safe_obs = []
        candidates = random.sample(config.obstacle_position, len(config.obstacle_position))
        JITTER = 1.0
        MIN_DIST_GOAL = 2.5
        MIN_DIST_START = 2.0

        for pos in candidates:
            if len(safe_obs) >= 10:
                break
            placed = False
            for _ in range(20):
                jx = random.uniform(-JITTER, JITTER)
                jy = random.uniform(-JITTER, JITTER)
                nx = pos[0] + jx
                ny = pos[1] + jy
                
                d_start = math.hypot(nx - self.start[0][0], ny - self.start[0][1])
                d_goal = math.hypot(nx - self.goal[0][0], ny - self.goal[0][1])

                if d_start >= MIN_DIST_START and d_goal >= MIN_DIST_GOAL:
                    # Chiều cao ngẫu nhiên trong khoảng (0.1, 3.0)m
                    h_rand = random.uniform(config.MIN_OBSTACLE_HEIGHT, config.MAX_OBSTACLE_HEIGHT)
                    safe_obs.append([nx, ny, h_rand])
                    placed = True
                    break
            
            if not placed:
                d_start = math.hypot(pos[0] - self.start[0][0], pos[1] - self.start[0][1])
                d_goal = math.hypot(pos[0] - self.goal[0][0], pos[1] - self.goal[0][1])
                if d_start >= MIN_DIST_START and d_goal >= MIN_DIST_GOAL:
                    h_rand = random.uniform(config.MIN_OBSTACLE_HEIGHT, config.MAX_OBSTACLE_HEIGHT)
                    safe_obs.append([pos[0], pos[1], h_rand])

        return safe_obs

    def update_mujoco_bodies(self):
        """Cập nhật tọa độ và kích thước chiều cao của 10 cột trong MuJoCo."""
        for i, obs in enumerate(self.obstacle_data):
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_obstacle_{i}")
            
            h_total = obs[2]
            half_h = h_total / 2.0
            
            if b_id >= 0:
                self.model.body_pos[b_id] = [obs[0], obs[1], half_h]
            if g_id >= 0:
                self.model.geom_size[g_id][1] = half_h

        # Cột đích màu đỏ
        g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_pole")
        if g_id >= 0:
            self.model.body_pos[g_id] = [self.goal[0][0], self.goal[0][1], 1.5]

        mujoco.mj_forward(self.model, self.data)

    def _observationSpace(self):
        """Quan sát là 1 ảnh RGB kích thước (64, 64, 3) từ camera FPV của drone."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            dtype=np.uint8
        )

    def _actionSpace(self):
        """Không gian hành động chuẩn hóa [-1.0, 1.0] cho PPO (Stable-Baselines3):
        - action[0] in [-1.0, 1.0]: ánh xạ sang alpha in [0°, 180°] (0.0 là bay thẳng 90°)
        - action[1] in [-1.0, 1.0]: ánh xạ sang beta  in [0°, 180°] (0.0 là bay ngang 90°)
        - action[2] in [-1.0, 1.0]: ánh xạ sang d     in [0.0, 2.0]m (0.0 là tiến tới 1.0m)
        """
        return spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_fpv_image(self):
        """Lấy ảnh RGB (64, 64, 3) từ camera FPV trên drone."""
        imgs = self._getDroneImages(0)
        rgb = imgs[0]
        if rgb is not None:
            return rgb[:, :, :3].astype(np.uint8)
        return np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)

    def _computeObs(self):
        return self._get_fpv_image()

    def _go_up_hover(self, target_z=1.0, timeout=5.0):
        """Cất cánh ban đầu lên độ cao ổn định trước khi giao quyền cho PPO."""
        self.control.reset()
        cur_pos = self.pos[0]
        target = np.array([cur_pos[0], cur_pos[1], target_z])
        start_t = time.time()

        for _ in range(int(self.CTRL_FREQ * timeout)):
            cur_pos = self.pos[0]
            cur_quat = self.quat[0]
            cur_vel = self.vel[0]
            cur_ang_v = self.ang_v[0]

            rpm, _, _ = self.control.computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_v,
                target_pos=target,
                target_rpy=np.array([0.0, 0.0, self.rpy[0, 2]])
            )
            super().step(rpm.flatten())
            if self.GUI:
                self.render()

            if abs(cur_pos[2] - target_z) < 0.08 and np.linalg.norm(cur_vel) < 0.2:
                break

    def reset(self, seed=None, options=None):
        """Reset môi trường: tạo chướng ngại vật ngẫu nhiên và đưa drone lên vị trí hover."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0
        self.collision = False
        self.win = False
        self.over_map = False

        self.start, self.goal = self._sample_start_goal()
        self.INIT_XYZS = self.start
        self.obstacle_data = self._generate_safe_obstacles()

        # Reset dữ liệu MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Đặt lại vị trí ban đầu cho drone
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_addr:qpos_addr + 3] = self.start[0]
        # Yaw ban đầu nhìn theo chiều -Y (tương đương -90 độ)
        cr, sr = np.cos(0), np.sin(0)
        cp, sp = np.cos(0), np.sin(0)
        cy, sy = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
        self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [cy, 0, 0, sy]

        self.update_mujoco_bodies()
        self._updateAndStoreKinematicInformation()

        # Cất cánh lên 1.0m an toàn
        self._go_up_hover(target_z=1.0)
        self.previous_dist = np.linalg.norm(self.pos[0] - self.goal[0])

        obs = self._get_fpv_image()
        info = self._computeInfo()
        return obs, info

    def step(self, action):
        """Thực thi hành động cấp cao từ PPO bằng vòng lặp PID cấp thấp."""
        act_arr = np.array(action, dtype=np.float32).flatten()

        # 1. Giải mã action:
        # Nếu action truyền vào dạng độ thuần túy (có giá trị > 1.0 hoặc < -1.0)
        if np.any(act_arr > 1.0) or np.any(act_arr < -1.0):
            alpha = float(np.clip(act_arr[0], 0.0, 180.0))
            beta = float(np.clip(act_arr[1], 0.0, 180.0))
            d = float(np.clip(act_arr[2], 0.0, config.MAX_STEP_DISTANCE))
        else:
            # Chuẩn hóa PPO [-1.0, 1.0]:
            # act=0.0 -> alpha=90° (bay thẳng), beta=90° (giữ độ cao), d=1.0m (tiến tới 1m)
            a = np.clip(act_arr, -1.0, 1.0)
            alpha = float((a[0] + 1.0) * 0.5 * 180.0)
            beta = float((a[1] + 1.0) * 0.5 * 180.0)
            d = float((a[2] + 1.0) * 0.5 * config.MAX_STEP_DISTANCE)

        # 2. Đổi sang radian
        alpha_rad = np.radians(alpha)
        # Cách 2: beta có tâm 90° là bay ngang, <90° chúc xuống, >90° bay lên
        theta_pitch = np.radians(beta - 90.0)

        # 3. Tính toán hình chiếu theo công thức lượng giác (Góc nhìn FPV)
        r_xy = d * np.cos(theta_pitch)
        dx_formula = r_xy * np.cos(alpha_rad)  # Lệch sang phải/trái FPV
        dy_formula = r_xy * np.sin(alpha_rad)  # Tiến thẳng theo FPV
        dz = d * np.sin(theta_pitch)

        # 4. Chuyển đổi từ hệ quy chiếu Camera/Body sang Tọa độ thế giới (World Frame)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        rot_mat = np.array(self.data.xmat[body_id]).reshape(3, 3)
        v_forward = rot_mat[:, 0]  # Trục X body: hướng tiến tới trước của camera
        v_right = rot_mat[:, 1]    # Trục Y body: hướng sang phải của camera

        cur_pos = self.pos[0].copy()
        delta_world_xy = dy_formula * v_forward[:2] + dx_formula * v_right[:2]

        x1 = cur_pos[0] + delta_world_xy[0]
        y1 = cur_pos[1] + delta_world_xy[1]
        
        # 5. Chặn cứng chống xuyên đất (Hard clamp: [MIN_Z, MAX_Z])
        z1 = float(np.clip(cur_pos[2] + dz, config.MIN_Z, config.MAX_Z))

        self.target_pos = np.array([x1, y1, z1])
        
        # Chỉ quay mũi nếu quãng đường dịch chuyển đủ lớn (d >= 0.15m) để tránh drone xoay tít tại chỗ
        if d >= 0.15 and np.linalg.norm(delta_world_xy) > 1e-3:
            target_yaw = math.atan2(delta_world_xy[1], delta_world_xy[0])
        else:
            target_yaw = self.rpy[0, 2]
        self.target_rpy = np.array([0.0, 0.0, target_yaw])

        # 6. Vòng lặp PID cấp thấp bám theo mục tiêu (tương tự hàm move() cũ)
        substep_accum_reward = 0.0
        
        for sub in range(config.SUBSTEPS_PER_ACTION):
            cur_p = self.pos[0]
            cur_q = self.quat[0]
            cur_v = self.vel[0]
            cur_w = self.ang_v[0]

            rpm, _, _ = self.control.computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=cur_p,
                cur_quat=cur_q,
                cur_vel=cur_v,
                cur_ang_vel=cur_w,
                target_pos=self.target_pos,
                target_rpy=self.target_rpy
            )

            super().step(rpm.flatten())
            if self.GUI:
                self.render()

            # Tính điểm thưởng từng bước phụ
            sub_r = self._computeSubstepReward()
            substep_accum_reward += sub_r

            # Kiểm tra va chạm hoặc kết thúc sớm
            if self.collision or self.win or self.over_map:
                break

            # Điều kiện dừng sớm khi đã bám sát tọa độ đích
            dist_to_waypoint = np.linalg.norm(self.pos[0] - self.target_pos)
            speed = np.linalg.norm(self.vel[0])
            wobble = np.linalg.norm(self.ang_v[0])
            if sub > 20 and dist_to_waypoint < 0.08 and speed < 0.15 and wobble < 0.2:
                break

        # 7. Cập nhật biến đếm và điểm thưởng
        self.step_count += 1
        self.accumulated_substep_reward = substep_accum_reward
        self.total_accumulated_reward += substep_accum_reward

        obs = self._get_fpv_image()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, float(substep_accum_reward), terminated, truncated, info

    def _computeSubstepReward(self):
        """Tính điểm thưởng cho mỗi bước điều khiển PID con."""
        cur_p = self.pos[0]
        cur_dist_goal = np.linalg.norm(cur_p - self.goal[0])
        progress = self.previous_dist - cur_dist_goal
        self.previous_dist = cur_dist_goal

        r = 3.0 * progress - 0.01  # Thưởng tiến tới đích + Phạt thời gian nhẹ

        # Kiểm tra khoảng cách gần cột vật cản
        min_obstacle_dist = self._get_min_obstacle_distance()
        if min_obstacle_dist < 1.75:
            # Phạt khi tiến quá gần vào vùng nguy hiểm quanh cột
            r -= 0.05 * (1.75 - min_obstacle_dist) / 1.75

        # Kiểm tra va chạm, vượt map, tới đích
        self.collision = self.check_collision()
        self.win = self.check_win()
        self.over_map = self.check_overmap()

        if self.collision:
            r -= 200.0
        if self.over_map:
            r -= 200.0
        if self.win:
            r += 500.0

        return r

    def _get_min_obstacle_distance(self):
        """Tính khoảng cách ngắn nhất từ drone đến bề mặt các cột vật cản (có xét chiều cao cột)."""
        drone_x, drone_y, drone_z = self.pos[0]
        min_d = 999.0

        for obs in self.obstacle_data:
            ox, oy, h_total = obs
            # Nếu drone đang bay thấp hơn đỉnh cột
            if drone_z <= h_total:
                d_center = math.hypot(drone_x - ox, drone_y - oy)
                d_surface = max(0.0, d_center - config.CYLINDER_RADIUS)
                if d_surface < min_d:
                    min_d = d_surface
        return min_d

    def check_collision(self):
        drone_x, drone_y, drone_z = self.pos[0]
        if drone_z < 0.05:
            return True

        # Ngưỡng va chạm theo yêu cầu: lại gần cột trong khoảng 0.3m
        threshold = config.CYLINDER_RADIUS + config.COLLISION_DISTANCE

        for obs in self.obstacle_data:
            ox, oy, h_total = obs
            # Va chạm xảy ra khi độ cao drone <= chiều cao đỉnh cột
            if drone_z <= h_total + 0.05:
                d_center = math.hypot(drone_x - ox, drone_y - oy)
                if d_center <= threshold:
                    return True
        return False

    def check_win(self):
        cur_x, cur_y = self.pos[0][:2]
        goal_x, goal_y = self.goal[0][:2]
        return abs(goal_x - cur_x) <= 1.0 and abs(goal_y - cur_y) <= 1.0

    def check_overmap(self):
        x, y = self.pos[0][:2]
        return abs(x) > config.MAP_LIMIT_X or abs(y) > config.MAP_LIMIT_Y

    def _computeTerminated(self):
        return bool(self.collision or self.over_map)

    def _computeTruncated(self):
        # Truncated nếu đi quá 70 bước cấp cao hoặc đã tới đích
        return bool(self.step_count >= config.MAX_STEPS or self.win)

    def _computeInfo(self):
        cur_dist = float(np.linalg.norm(self.pos[0] - self.goal[0]))
        return {
            "is_success": bool(self.win),
            "collision": bool(self.collision),
            "over_map": bool(self.over_map),
            "distance_to_goal": cur_dist,
            "step_count": int(self.step_count),
            "target_pos": self.target_pos.tolist(),
            "accumulated_substep_reward": float(self.accumulated_substep_reward),
            "total_accumulated_reward": float(self.total_accumulated_reward)
        }
