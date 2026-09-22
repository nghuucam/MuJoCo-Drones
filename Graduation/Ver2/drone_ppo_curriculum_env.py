import os
import sys
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
import config


class DronePPOCurriculumEnv(BaseAviary):
    """Môi trường Gymnasium nâng cấp cho Drone PPO tích hợp Curriculum Learning & Cột trụ Gai xương rồng.
    
    Tính năng nổi bật trong Ver2:
    1. Curriculum Learning (4 Cấp độ): Tự động mở rộng số lượng vật cản (3 -> 5 -> 7 -> 10) 
       và đẩy vị trí Đích xa dần nhưng giữ nằm trong khoảng Y >= 0.
    2. Cột trụ Gai xương rồng (Cactus-like Spikes): Mỗi cột trụ được gắn ngẫu nhiên từ 3 đến 6 gai nhọn
       (độ dài 0.2 - 0.8m, bán kính 0.05 - 0.20m, góc xoay & độ cao ngẫu nhiên).
    3. Hệ số va chạm mới: COLLISION_MARGIN = 0.05m giúp drone lách sát sạt qua các ngọn gai mượt mà.
    """

    def __init__(self, gui: bool = False):
        self.gui_mode = gui
        self.current_level = 0  # Level 0 đến Level 3

        # Đặt trước một mục tiêu tạm thời để khởi tạo XML ban đầu
        self.goal = np.array([[0.0, 7.5, 1.5]])
        self.obstacle_data = []  # Lưu danh sách [ox, oy, h_total, spikes_info]

        xml_str = self._generate_cactus_xml(
            start_pos=[0.0, 11.0, 1.0],
            goal_pos=self.goal[0],
            level=self.current_level
        )

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            physics=Physics.MJC,
            sim_freq=240,
            ctrl_freq=48,
            gui=gui,
            record=False,
            obstacles=False,
            obs_type=ObservationType.RGB,
            act_type=ActionType.RPM,
            vision_attributes=True,
            initial_xyzs=np.array([[0.0, 11.0, 1.0]]),
            initial_rpys=np.array([[0.0, 0.0, -math.pi / 2]])
        )

        # Định nghĩa không gian hành động chuẩn hóa [-1.0, 1.0] cho 3 tham số (alpha, beta, d)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Không gian quan sát ảnh camera FPV RGB (64 x 64 x 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8
        )

        # Bộ điều khiển PID tầng thấp (DSLPIDControl)
        self.control = DSLPIDControl(env=self)

        # Các biến trạng thái episode
        self.step_count = 0
        self.previous_dist = 0.0
        self.target_pos = np.array([0.0, 11.0, 1.0])
        self.target_rpy = np.array([0.0, 0.0, -math.pi / 2])
        self.collision = False
        self.win = False
        self.over_map = False

        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0

    def set_level(self, level: int):
        """Cập nhật cấp độ Curriculum (Level 0 đến 3)."""
        self.current_level = int(np.clip(level, 0, 3))

    def reset(self, seed=None, options=None):
        """Khởi tạo lại môi trường cho episode mới dựa trên level hiện tại."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 1. Chọn ngẫu nhiên vị trí xuất phát từ config.start_space
        start_idx = random.randint(0, len(config.start_space) - 1)
        start_xy = config.start_space[start_idx]
        start_pos = np.array([start_xy[0], start_xy[1], 1.0])

        # 2. Chọn vị trí Đích đến theo khoảng Y của Level hiện tại (không vượt quá Y = 0)
        y_min, y_max = config.GOAL_Y_RANGES[self.current_level]
        goal_x = float(np.random.uniform(-2.0, 2.0))
        goal_y = float(np.random.uniform(y_min, y_max))
        self.goal = np.array([[goal_x, goal_y, 1.5]])

        # 3. Sinh XML mới cho MuJoCo với cây xương rồng & cấp độ tương ứng
        xml_str = self._generate_cactus_xml(
            start_pos=start_pos,
            goal_pos=self.goal[0],
            level=self.current_level
        )

        # Nạp lại mô hình MuJoCo linh hoạt từ chuỗi XML vừa sinh
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # Thiết lập lại vị trí xuất phát ban đầu trong MuJoCo Data
        self.INIT_XYZS = np.array([start_pos])
        self.INIT_RPYS = np.array([[0.0, 0.0, -math.pi / 2]])
        
        # Gọi reset của lớp cha
        obs, info = super().reset(seed=seed)

        # Reset bộ điều khiển PID
        self.control.reset()

        # Reset các cờ và biến đếm
        self.step_count = 0
        self.target_pos = start_pos.copy()
        self.target_rpy = np.array([0.0, 0.0, -math.pi / 2])
        self.previous_dist = np.linalg.norm(start_pos - self.goal[0])
        self.collision = False
        self.win = False
        self.over_map = False
        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0

        # Lấy ảnh FPV đầu tiên
        fpv_obs = self._get_fpv_image()
        info["curriculum_level"] = self.current_level
        info["is_success"] = False
        info["success"] = False

        return fpv_obs, info

    def step(self, action):
        """Thực hiện 1 bước hành động cấp cao PPO (xuất góc cầu alpha, beta và khoảng cách d)."""
        # 1. Giải mã hành động từ [-1, 1]^3 sang (alpha, beta, d)
        norm_a, norm_b, norm_d = action[0], action[1], action[2]

        # Góc phương vị alpha in [0 deg, 180 deg] (Góc quay theo chiều ngang)
        alpha_deg = float((norm_a + 1.0) * 90.0)
        # Góc nâng beta in [70 deg, 110 deg] (Bay ngang ổn định, biên độ cao z tối đa +-0.3m/step)
        beta_deg = float(70.0 + (norm_b + 1.0) * 20.0)
        # Khoảng cách bước d in [0.1m, 2.0m]
        d = float(0.1 + (norm_d + 1.0) * 0.95)

        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)

        # 2. Tính vector chuyển dời spherical trong hệ tọa độ địa phương của drone
        dx_local = d * math.sin(beta_rad) * math.cos(alpha_rad)
        dy_local = d * math.sin(beta_rad) * math.sin(alpha_rad)
        dz_local = d * math.cos(beta_rad)

        # 3. Chuyển sang hệ tọa độ thế giới (World Frame) dựa theo Yaw hiện tại
        current_yaw = self.rpy[0, 2]
        delta_world_x = dx_local * math.cos(current_yaw) - dy_local * math.sin(current_yaw)
        delta_world_y = dx_local * math.sin(current_yaw) + dy_local * math.cos(current_yaw)
        delta_world_z = dz_local

        # 4. Tính tọa độ mục tiêu mới cho PID
        raw_target_x = self.pos[0, 0] + delta_world_x
        raw_target_y = self.pos[0, 1] + delta_world_y
        raw_target_z = self.pos[0, 2] + delta_world_z

        # 5. Khống chế an toàn cứng (Hard Bounds Clamping)
        target_x = np.clip(raw_target_x, -config.MAP_LIMIT_X, config.MAP_LIMIT_X)
        target_y = np.clip(raw_target_y, -config.MAP_LIMIT_Y, config.MAP_LIMIT_Y)
        target_z = np.clip(raw_target_z, config.MIN_Z, config.MAX_Z)

        self.target_pos = np.array([target_x, target_y, target_z])

        # Hướng quay target_yaw (Chỉ quay khi dịch chuyển đủ xa)
        if d >= 0.15 and np.hypot(delta_world_x, delta_world_y) > 1e-3:
            target_yaw = math.atan2(delta_world_y, delta_world_x)
        else:
            target_yaw = current_yaw
        self.target_rpy = np.array([0.0, 0.0, target_yaw])

        # 6. Vòng lặp điều khiển PID con tầng thấp (Up to 150 substeps)
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
            if self.gui_mode:
                self.render()

            sub_r = self._computeSubstepReward()
            substep_accum_reward += sub_r

            if self.collision or self.win or self.over_map:
                break

            # Ngắt sớm khi đã bám sát điểm mục tiêu trung gian
            dist_to_wp = np.linalg.norm(self.pos[0] - self.target_pos)
            speed = np.linalg.norm(self.vel[0])
            if sub > 20 and dist_to_wp < 0.08 and speed < 0.15:
                break

        # 7. Cập nhật đếm bước và thông số trả về
        self.step_count += 1
        self.accumulated_substep_reward = substep_accum_reward
        self.total_accumulated_reward += substep_accum_reward

        obs = self._get_fpv_image()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, float(substep_accum_reward), terminated, truncated, info

    def _computeSubstepReward(self):
        """Hàm tính điểm thưởng cải tiến cho Ver2 (Phạt sát gai + Thưởng tiến bộ + Phạt thời gian)."""
        cur_p = self.pos[0]
        cur_dist_goal = np.linalg.norm(cur_p - self.goal[0])
        progress = self.previous_dist - cur_dist_goal
        self.previous_dist = cur_dist_goal

        # 1. Thưởng tiến về phía đích + Phạt thời gian bước
        r = 3.5 * progress - 0.01

        # 2. Phạt nguy hiểm khi tiến gần cột trụ và các gai nhọn (khoảng cách < 1.5m)
        min_obstacle_dist = self._get_min_obstacle_distance()
        if min_obstacle_dist < 1.5:
            r -= 0.08 * (1.5 - min_obstacle_dist) / 1.5

        # 3. Kiểm tra va chạm & sự kiện
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

    def check_collision(self):
        """Kiểm tra va chạm vật lý trực tiếp từ MuJoCo hoặc khoảng cách vi phạm margin 0.05m."""
        # 1. Kiểm tra va chạm va quệt vật lý thực tế từ bộ giải MuJoCo
        if self.data.ncon > 0:
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                g1 = self.model.geom(contact.geom1).name
                g2 = self.model.geom(contact.geom2).name
                if "arrow" in g1 or "arrow" in g2:
                    continue  # Bỏ qua con trỏ mũi tên trang trí
                if "drone0" in g1 or "drone0" in g2:
                    if "obstacle" in g1 or "obstacle" in g2 or "floor" in g1 or "floor" in g2:
                        return True

        # 2. Kiểm tra va chạm hình học với biên lề an toàn COLLISION_MARGIN = 0.05m
        min_d = self._get_min_obstacle_distance()
        if min_d <= config.COLLISION_MARGIN:
            return True

        return False

    def check_win(self):
        """Kiểm tra drone đã bay vào vùng đích an toàn hay chưa."""
        dist = np.linalg.norm(self.pos[0] - self.goal[0])
        return dist < 0.85

    def check_overmap(self):
        """Kiểm tra drone có bay chệch khỏi bản đồ không."""
        px, py, pz = self.pos[0]
        if abs(px) > config.MAP_LIMIT_X + 0.5 or abs(py) > config.MAP_LIMIT_Y + 0.5:
            return True
        if pz < 0.25 or pz > config.MAX_Z + 0.5:
            return True
        return False

    def _computeTerminated(self):
        return self.collision or self.win or self.over_map

    def _computeTruncated(self):
        return self.step_count >= config.MAX_STEPS

    def _computeInfo(self):
        is_succ = bool(self.win and not self.collision)
        return {
            "is_success": is_succ,
            "success": is_succ,
            "step_count": int(self.step_count),
            "collision": bool(self.collision),
            "win": bool(self.win),
            "curriculum_level": int(self.current_level),
            "accumulated_reward": float(self.accumulated_substep_reward),
            "total_reward": float(self.total_accumulated_reward)
        }

    def _get_min_obstacle_distance(self):
        """Tính khoảng cách ngắn nhất từ drone đến bề mặt cột trụ VÀ bề mặt các gai xương rồng."""
        drone_p = self.pos[0]
        min_d = 999.0

        for obs in self.obstacle_data:
            ox, oy, h_total, spikes = obs
            # 1. Khoảng cách đến thân cột trụ chính
            if drone_p[2] <= h_total:
                d_center = math.hypot(drone_p[0] - ox, drone_p[1] - oy)
                d_surf = max(0.0, d_center - config.CYLINDER_RADIUS)
                if d_surf < min_d:
                    min_d = d_surf

            # 2. Khoảng cách đến đầu ngọn của từng gai xương rồng
            for (sp_x, sp_y, sp_z, sp_len, sp_rad) in spikes:
                # Vị trí ngọn gai trong thế giới
                tip_x = ox + sp_x
                tip_y = oy + sp_y
                tip_z = sp_z
                dist_tip = np.linalg.norm(drone_p - np.array([tip_x, tip_y, tip_z]))
                dist_spike_surf = max(0.0, dist_tip - sp_rad)
                if dist_spike_surf < min_d:
                    min_d = dist_spike_surf

        return min_d

    def _get_fpv_image(self):
        """Trích xuất ảnh quan sát FPV (64x64 RGB) từ camera gắn trên mũi drone."""
        rgb_img = self._get_drone_camera_image(drone_idx=0, camera_name="drone0_cam")
        return rgb_img

    def _get_drone_camera_image(self, drone_idx=0, camera_name="drone0_cam"):
        """Render ảnh camera offscreen thông qua MuJoCo Renderer."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)

        renderer = mujoco.Renderer(self.model, height=config.IMG_HEIGHT, width=config.IMG_WIDTH)
        renderer.update_scene(self.data, camera=cam_id)
        rgb = renderer.render()
        renderer.close()
        return rgb

    def _generate_cactus_xml(self, start_pos, goal_pos, level):
        """Sinh chuỗi XML mô phỏng MuJoCo linh hoạt với các cột trụ có gai xương rồng ngẫu nhiên."""
        num_obstacles = config.OBSTACLE_COUNTS.get(level, 3)
        self.obstacle_data = []

        obstacle_bodies_xml = ""

        # Sinh XML cho các cột trụ được kích hoạt theo level
        for i in range(num_obstacles):
            base_pos = config.OBSTACLE_POSITIONS[i]
            ox, oy = base_pos[0], base_pos[1]
            h_total = float(np.random.uniform(config.MIN_OBSTACLE_HEIGHT, config.MAX_OBSTACLE_HEIGHT))
            h_half = h_total / 2.0

            # Sinh số gai ngẫu nhiên từ 3 đến 6 cho cột này
            num_spikes = random.randint(config.SPIKES_PER_PILLAR_MIN, config.SPIKES_PER_PILLAR_MAX)
            spikes_list = []
            spikes_xml = ""

            for sp_idx in range(num_spikes):
                # Độ cao z ngẫu nhiên dọc theo thân cột
                sp_z = float(np.random.uniform(0.3, h_total - 0.2))
                # Góc hướng gai ra ngoài theta in [0, 2pi]
                theta = float(np.random.uniform(0, 2 * math.pi))
                # Bán kính & độ dài gai
                sp_rad = float(np.random.uniform(config.SPIKE_RADIUS_MIN, config.SPIKE_RADIUS_MAX))
                sp_len = float(np.random.uniform(config.SPIKE_LENGTH_MIN, config.SPIKE_LENGTH_MAX))

                # Vector hướng ngọn gai nhô ra từ tâm cột
                dir_x = math.cos(theta)
                dir_y = math.sin(theta)
                dir_z = float(np.random.uniform(-0.1, 0.2))  # Hơi chếch lên hoặc ngang

                # Vị trí ngọn gai tương đối so với tâm cột
                tip_rel_x = (config.CYLINDER_RADIUS + sp_len) * dir_x
                tip_rel_y = (config.CYLINDER_RADIUS + sp_len) * dir_y

                spikes_list.append((tip_rel_x, tip_rel_y, sp_z, sp_len, sp_rad))

                # Vị trí tâm geom gai trong không gian body cột trụ
                spike_pos_z = sp_z - h_half
                spike_center_x = (config.CYLINDER_RADIUS + sp_len / 2.0) * dir_x
                spike_center_y = (config.CYLINDER_RADIUS + sp_len / 2.0) * dir_y

                spikes_xml += f"""
        <geom name="obstacle_{i}_spike_{sp_idx}" type="cylinder" 
              pos="{spike_center_x:.3f} {spike_center_y:.3f} {spike_pos_z:.3f}" 
              zaxis="{dir_x:.3f} {dir_y:.3f} {dir_z:.3f}" 
              size="{sp_rad:.3f} {sp_len / 2.0:.3f}" 
              rgba="0.85 0.45 0.1 1" contype="1" conaffinity="1"/>"""

            self.obstacle_data.append((ox, oy, h_total, spikes_list))

            obstacle_bodies_xml += f"""
    <body name="obstacle_{i}" pos="{ox} {oy} {h_half}">
      <geom name="geom_obstacle_{i}" type="cylinder" size="{config.CYLINDER_RADIUS} {h_half}" rgba="0.1 0.6 0.25 1" contype="1" conaffinity="1"/>{spikes_xml}
    </body>"""

        # Cột mốc Đích màu đỏ
        goal_x, goal_y = goal_pos[0], goal_pos[1]
        goal_body_xml = f"""
    <body name="goal_pole" pos="{goal_x} {goal_y} 1.5">
      <geom type="cylinder" size="0.15 1.5" rgba="1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 1.6" size="0.4" rgba="1 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>"""

        # Trích xuất XML mô hình Drone CF2X
        drone_x, drone_y, drone_z = start_pos[0], start_pos[1], start_pos[2]
        drone_xml = f"""
    <body name="drone0" pos="{drone_x} {drone_y} {drone_z}" quat="0.707 0 0 -0.707">
      <freejoint name="drone0_joint"/>
      <inertial pos="0 0 0" mass="0.027" diaginertia="1.4e-5 1.4e-5 2.1e-5"/>
      <geom name="drone0_collision" type="cylinder" size="0.06 0.025" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom name="drone0_arrow_stem" type="cylinder" pos="0 0 1.8" size="0.12 0.7" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom name="drone0_arrow_pointer" type="sphere" pos="0 0 0.8" size="0.4" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <site name="drone0_center" pos="0 0 0" group="5"/>
      <site name="drone0_prop0" pos="0.028 0.028 0" group="5"/>
      <site name="drone0_prop1" pos="-0.028 0.028 0" group="5"/>
      <site name="drone0_prop2" pos="-0.028 -0.028 0" group="5"/>
      <site name="drone0_prop3" pos="0.028 -0.028 0" group="5"/>
      <camera name="drone0_cam" pos="0.03 0 0.01" xyaxes="0 -1 0 0 0 1" fovy="75"/>
    </body>"""

        sensors_xml = """
    <gyro name="drone0_gyro" site="drone0_center"/>
    <accelerometer name="drone0_acc" site="drone0_center"/>
    <framequat name="drone0_quat" objtype="site" objname="drone0_center"/>
    <framepos name="drone0_pos" objtype="site" objname="drone0_center"/>
    <framelinvel name="drone0_vel" objtype="site" objname="drone0_center"/>
    <frameangvel name="drone0_angvel" objtype="site" objname="drone0_center"/>"""

        full_xml = f"""<mujoco model="drone_ppo_curriculum">
  <option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="0.00416666"/>
  <compiler inertiafromgeom="false" autolimits="true"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="15 15 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_xml}
{obstacle_bodies_xml}
{goal_body_xml}
  </worldbody>

  <sensor>
{sensors_xml}
  </sensor>
</mujoco>"""
        return full_xml
