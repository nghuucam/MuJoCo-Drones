import os
import sys
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from multi_drone_mujoco.envs.base_aviary import BaseAviary, CF2_MESH_DIR
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
import config


class DronePPOCurriculumEnv(BaseAviary):
    """Môi trường Gymnasium nâng cấp cho Drone PPO tích hợp Curriculum Learning & Cột trụ Gai xương rồng (Ver4).
    
    Tính năng nổi bật trong Ver4:
    1. Curriculum Learning (4 Cấp độ): Tăng gấp đôi độ dài bản đồ trục Y cho Level 1->3 (lên tới 21m).
    2. Cột trụ Gai xương rồng: Tăng số lượng vật cản (Level 6 lên tới 15 cột) 
       và tăng mật độ số lượng gai trên mỗi cột (10 đến 16 gai, dài 0.30 - 1.10m).
    3. Tọa độ đích: Thu hẹp bán kính đích GOAL_THRESHOLD xuống 0.5m (đòi hỏi drone bay 3D chuẩn xác).
    4. Không gian hành động: Khống chế góc lái alpha in [-35, +35] deg và beta in [-30, +30] deg nằm trọn trong camera FOV 75 deg.
    """

    def __init__(self, gui: bool = False, show_marker: bool = True):
        self.gui_mode = gui
        self.show_marker = show_marker
        self.current_level = 0

        self.goal = np.array([[0.0, 7.5, 1.5]])
        self.obstacle_data = []

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
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            vision_attributes=True,
            initial_xyzs=np.array([[0.0, 11.0, 1.0]]),
            initial_rpys=np.array([[0.0, 0.0, -math.pi / 2]])
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0, high=255, shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-50.0, high=50.0, shape=(18,), dtype=np.float32
            )
        })

        self.control = DSLPIDControl(env=self)

        self.step_count = 0
        self.previous_dist = 0.0
        self.best_dist_to_goal = 999.0
        self.target_pos = np.array([0.0, 11.0, 1.0])
        self.target_rpy = np.array([0.0, 0.0, -math.pi / 2])
        self.collision = False
        self.win = False
        self.over_map = False
        self.cleared_obstacles = set()

        self.x_min_limit = -7.0
        self.x_max_limit = 7.0
        self.y_min_limit = -12.5
        self.y_max_limit = 12.5

        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0

        self.saved_cam_azimuth = getattr(config, "GUI_CAMERA_AZIMUTH", -90.0)
        self.saved_cam_elevation = getattr(config, "GUI_CAMERA_ELEVATION", -20.0)
        self.saved_cam_distance = getattr(config, "GUI_CAMERA_DISTANCE", 3.5)
        self.saved_cam_lookat = np.array([0.0, 11.0, 1.0], dtype=float)
        self.has_user_camera = False

        self._fpv_window = None
        self._fpv_label = None
        self._fpv_img_tk = None

    def set_level(self, level: int):
        """Cập nhật cấp độ Curriculum (Level 0 đến 6)."""
        max_level = len(config.GOAL_Y_RANGES) - 1
        self.current_level = int(np.clip(level, 0, max_level))

    def reset(self, seed=None, options=None):
        """Khởi tạo lại môi trường cho episode mới dựa trên level hiện tại."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        start_idx = random.randint(0, len(config.start_space) - 1)
        start_xy = config.start_space[start_idx]
        start_pos = np.array([start_xy[0], start_xy[1], 1.0])

        y_min, y_max = config.GOAL_Y_RANGES[self.current_level]
        goal_x = float(np.random.uniform(-2.0, 2.0))
        goal_y = float(np.random.uniform(y_min, y_max))
        self.goal = np.array([[goal_x, goal_y, 1.5]])

        xml_str = self._generate_cactus_xml(
            start_pos=start_pos,
            goal_pos=self.goal[0],
            level=self.current_level
        )

        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        if hasattr(self, "_cam_renderer") and self._cam_renderer is not None:
            try:
                self._cam_renderer.close()
            except Exception:
                pass
            self._cam_renderer = None

        self.INIT_XYZS = np.array([start_pos])
        self.INIT_RPYS = np.array([[0.0, 0.0, -math.pi / 2]])
        
        obs, info = super().reset(seed=seed)

        if self.gui_mode and hasattr(self, "_viewer") and self._viewer is not None:
            if self._viewer.is_running():
                try:
                    self._capture_camera_settings()
                    sim = self._viewer._get_sim()
                    if sim is not None:
                        sim.load(self.model, self.data, "")
                        sim.ui1_enable = getattr(config, "GUI_SHOW_RIGHT_UI", False)
                        sim.ui0_enable = getattr(config, "GUI_SHOW_LEFT_UI", False)
                        self._apply_camera_settings()
                        self._viewer.sync()
                except Exception:
                    self._viewer = None
            else:
                self._viewer = None

        self.control.reset()

        self.step_count = 0
        self.target_pos = start_pos.copy()
        self.target_rpy = np.array([0.0, 0.0, -math.pi / 2])
        self.previous_dist = np.linalg.norm(start_pos - self.goal[0])
        self.best_dist_to_goal = float(self.previous_dist)
        self.collision = False
        self.win = False
        self.over_map = False
        self.cleared_obstacles = set()
        self.accumulated_substep_reward = 0.0
        self.total_accumulated_reward = 0.0

        self.y_min_limit = float(self.goal[0][1] - config.OVERMAP_PAST_GOAL_DISTANCE)
        self.y_max_limit = float(start_pos[1] + config.OVERMAP_BEHIND_START_DISTANCE)

        obs_xs = [obs[0] for obs in self.obstacle_data]
        min_obs_x = min(obs_xs) if obs_xs else -5.0
        max_obs_x = max(obs_xs) if obs_xs else 5.0
        self.x_min_limit = float(min(min_obs_x - config.OVERMAP_X_MARGIN, start_pos[0] - 0.5, self.goal[0][0] - 0.5))
        self.x_max_limit = float(max(max_obs_x + config.OVERMAP_X_MARGIN, start_pos[0] + 0.5, self.goal[0][0] + 0.5))

        obs = self._get_obs()
        info["curriculum_level"] = self.current_level
        info["is_success"] = False
        info["success"] = False

        return obs, info

    def step(self, action):
        """Thực hiện 1 bước hành động cấp cao PPO (xuất góc cầu alpha, beta và khoảng cách d theo hệ Aviation Body Frame)."""
        norm_a = float(np.clip(action[0], -1.0, 1.0))
        norm_b = float(np.clip(action[1], -1.0, 1.0))
        norm_d = float(np.clip(action[2], -1.0, 1.0))

        max_alpha = float(getattr(config, "MAX_ALPHA_DEG", 35.0))
        max_beta = float(getattr(config, "MAX_BETA_DEG", 30.0))
        alpha_deg = float(norm_a * max_alpha)
        beta_deg = float(norm_b * max_beta)
        d = float(0.2 + (norm_d + 1.0) * 0.65)

        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)

        r_xy = d * math.cos(beta_rad)
        d_forward = r_xy * math.cos(alpha_rad)
        d_right = r_xy * math.sin(alpha_rad)
        dz = d * math.sin(beta_rad)

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        rot_mat = np.array(self.data.xmat[body_id]).reshape(3, 3)
        v_forward = rot_mat[:, 0]
        v_right = rot_mat[:, 1]

        delta_world_xy = d_forward * v_forward[:2] + d_right * v_right[:2]
        delta_world_x = float(delta_world_xy[0])
        delta_world_y = float(delta_world_xy[1])
        delta_world_z = float(dz)

        raw_target_x = self.pos[0, 0] + delta_world_x
        raw_target_y = self.pos[0, 1] + delta_world_y
        raw_target_z = self.pos[0, 2] + delta_world_z

        target_x = np.clip(raw_target_x, -config.MAP_LIMIT_X, config.MAP_LIMIT_X)
        target_y = np.clip(raw_target_y, -config.MAP_LIMIT_Y, config.MAP_LIMIT_Y)
        target_z = np.clip(raw_target_z, config.MIN_Z, config.MAX_Z)

        self.target_pos = np.array([target_x, target_y, target_z])

        if d >= 0.15 and np.hypot(delta_world_x, delta_world_y) > 1e-3:
            target_yaw = math.atan2(delta_world_y, delta_world_x)
        else:
            target_yaw = self.rpy[0, 2]
        self.target_rpy = np.array([0.0, 0.0, target_yaw])

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
                if hasattr(self, "_viewer") and self._viewer is not None and not self._viewer.is_running():
                    self.gui_mode = False
                elif sub % 2 == 0:
                    self.render()

            sub_r = self._computeSubstepReward()
            substep_accum_reward += sub_r

            if self.collision or self.win or self.over_map:
                break

            dist_to_wp = np.linalg.norm(self.pos[0] - self.target_pos)
            speed = np.linalg.norm(self.vel[0])
            if sub > 20 and dist_to_wp < 0.08 and speed < 0.15:
                break

        self.step_count += 1

        obs = self._get_obs()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()

        if truncated and not self.win:
            substep_accum_reward -= 50.0

        self.accumulated_substep_reward = substep_accum_reward
        self.total_accumulated_reward += substep_accum_reward

        info = self._computeInfo()

        return obs, float(substep_accum_reward), terminated, truncated, info

    def _computeSubstepReward(self):
        """Hàm tính điểm thưởng 8 thành phần chuẩn hóa tối ưu:
        1. R_progress: Thưởng tiến độ hướng đích (+1.0 * progress).
        2. R_milestone: Thưởng lách gai cột mốc (+5.0 chỉ 1 lần duy nhất khi vượt qua cột ở cự ly an toàn 0.1 - 0.6m).
        3. R_heading: Phạt bay ngược hướng đích (-0.01 * |cos(theta)| mỗi substep nếu cos(theta) < -0.2).
        4. R_idle: Phạt lười di chuyển có điều kiện (-0.005 mỗi substep nếu đường trống > 1.5m và tốc độ < 0.1m/s).
        5. R_proximity: Cảnh báo tiệm cận rủi ro sát gai (-0.005 mỗi substep nếu khoảng cách < 0.1m).
        6. R_altitude_boundary: Phạt tì đè biên sàn / trần (-0.01 mỗi substep nếu z <= MIN_Z+0.05 hoặc z >= MAX_Z-0.05).
        7. R_time: Chi phí thời gian (-0.002 mỗi substep).
        8. R_terminal: Thưởng chạm đích (+100.0) / Phạt va chạm, văng map (-50.0).
        """
        cur_p = self.pos[0]
        cur_dist_goal = float(np.linalg.norm(cur_p - self.goal[0]))

        r = -0.002

        if cur_dist_goal < self.best_dist_to_goal:
            progress = self.best_dist_to_goal - cur_dist_goal
            self.best_dist_to_goal = cur_dist_goal
            r += 1.0 * progress
        elif cur_dist_goal > self.previous_dist:
            regress = cur_dist_goal - self.previous_dist
            r -= 3.0 * regress

        self.previous_dist = cur_dist_goal

        for i, obs in enumerate(self.obstacle_data):
            if i not in self.cleared_obstacles:
                ox, oy, h_total, spikes = obs
                if cur_p[1] <= oy:
                    self.cleared_obstacles.add(i)
                    d_obs = self._get_distance_to_obstacle(i)
                    if 0.1 <= d_obs <= 0.6:
                        r += 5.0

        min_obstacle_dist = self._get_min_obstacle_distance()

        cur_v = self.vel[0]
        vec_to_goal = self.goal[0] - cur_p
        speed = float(np.linalg.norm(cur_v))
        dist_goal = float(cur_dist_goal)

        if speed > 0.05 and dist_goal > 1e-4:
            cos_theta = float(np.dot(cur_v, vec_to_goal) / (speed * dist_goal))
            if cos_theta < -0.2:
                r -= 0.01 * abs(cos_theta)

        if min_obstacle_dist > 1.5 and speed < 0.1:
            r -= 0.005

        if min_obstacle_dist < 0.1:
            r -= 0.005

        if cur_p[2] <= config.MIN_Z + 0.05 or cur_p[2] >= config.MAX_Z - 0.05:
            r -= 0.01

        self.collision = self.check_collision()
        self.win = self.check_win()
        self.over_map = self.check_overmap()

        if self.collision:
            r -= 50.0
        if self.over_map and not self.win:
            r -= 50.0
        if self.win:
            r += 100.0

        return r

    def check_collision(self):
        """Kiểm tra va chạm vật lý trực tiếp từ MuJoCo hoặc khoảng cách vi phạm margin 0.05m."""
        if self.data.ncon > 0:
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                g1 = self.model.geom(contact.geom1).name
                g2 = self.model.geom(contact.geom2).name
                if "arrow" in g1 or "arrow" in g2:
                    continue
                if "drone0" in g1 or "drone0" in g2:
                    if "obstacle" in g1 or "obstacle" in g2 or "floor" in g1 or "floor" in g2:
                        return True

        min_d = self._get_min_obstacle_distance()
        if min_d <= config.COLLISION_MARGIN:
            return True

        return False

    def check_win(self):
        """Kiểm tra drone đã bay vào vùng đích an toàn hay chưa."""
        dist = np.linalg.norm(self.pos[0] - self.goal[0])
        return bool(dist < getattr(config, "GOAL_THRESHOLD", 2.0))

    def check_overmap(self):
        """Kiểm tra drone có bay chệch khỏi bản đồ động (bóp theo đích và vị trí cột) không."""
        if self.win:
            return False

        px, py, pz = self.pos[0]

        if px < self.x_min_limit or px > self.x_max_limit:
            return True

        if py < self.y_min_limit or py > self.y_max_limit:
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
        over_step = bool(self.step_count >= config.MAX_STEPS and not self.win)
        return {
            "is_success": is_succ,
            "success": is_succ,
            "step_count": int(self.step_count),
            "collision": bool(self.collision),
            "win": bool(self.win),
            "over_map": bool(self.over_map),
            "over_step": over_step,
            "curriculum_level": int(self.current_level),
            "cleared_obstacles": int(len(self.cleared_obstacles)),
            "accumulated_reward": float(self.accumulated_substep_reward),
            "total_reward": float(self.total_accumulated_reward),
            "start_pos": self.INIT_XYZS[0].tolist(),
            "goal_pos": self.goal[0].tolist(),
            "drone_pos": self.pos[0].tolist()
        }

    def _get_distance_to_obstacle(self, obs_idx: int) -> float:
        """Tính khoảng cách ngắn nhất từ drone đến bề mặt cột trụ và các gai của cột trụ thứ obs_idx."""
        drone_p = self.pos[0]
        ox, oy, h_total, spikes = self.obstacle_data[obs_idx]
        d_center = math.hypot(drone_p[0] - ox, drone_p[1] - oy)
        d_surf = max(0.0, d_center - config.CYLINDER_RADIUS)
        if drone_p[2] > h_total:
            dz = drone_p[2] - h_total
            obs_d = math.hypot(d_surf, dz)
        else:
            obs_d = d_surf

        for (sp_x, sp_y, sp_z, sp_len, sp_rad) in spikes:
            tip_x = ox + sp_x
            tip_y = oy + sp_y
            tip_z = sp_z
            dist_tip = np.linalg.norm(drone_p - np.array([tip_x, tip_y, tip_z]))
            dist_spike_surf = max(0.0, dist_tip - sp_rad)
            if dist_spike_surf < obs_d:
                obs_d = dist_spike_surf

        return float(obs_d)

    def _get_min_obstacle_distance(self) -> float:
        """Tính khoảng cách ngắn nhất từ drone đến tất cả cột trụ VÀ bề mặt các gai xương rồng."""
        if not self.obstacle_data:
            return 999.0
        return float(min(self._get_distance_to_obstacle(i) for i in range(len(self.obstacle_data))))

    def _compute_apf_hint(self):
        """
        Tính toán gợi ý hành động an toàn dựa trên Trường thế nhân tạo (APF - Artificial Potential Field):
        1. Hướng mục tiêu (Attractive): Kéo Drone thẳng về vị trí Đích trên mặt phẳng ngang XY.
        2. Né hành lang va chạm (Anticipatory Corridor Deflection): Phát hiện cột cản trên đường đi và bẻ lái né trước.
        3. Phản xạ lực đẩy khẩn cấp sát bề mặt (Emergency Surface Repulsion): Đẩy lùi dạt ra xa nếu cự ly tới cột hoặc gai < 0.6m.
        4. Chuyển đổi sang hệ quy chiếu thân Drone (Body Frame) đồng bộ hoàn hảo với không gian hành động [-1.0, 1.0]:
           - alpha_safe_hint: Gợi ý góc lái [-1.0, 1.0] (tương ứng [-90 deg, +90 deg])
           - beta_safe_hint: Gợi ý góc ngẩng/chúc [-1.0, 1.0] (tương ứng [-30 deg, +30 deg])
        """
        cur_p = self.pos[0]
        cur_rpy = self.rpy[0]
        goal_p = self.goal[0]

        v_goal = (goal_p - cur_p)[:2]
        d_goal = float(np.linalg.norm(v_goal))
        if d_goal < 1e-4:
            return 0.0, 0.0
        u_goal = v_goal / d_goal

        r_corridor = config.CYLINDER_RADIUS + 0.35
        max_deflect_angle = 0.0
        best_deflect_dir = 0.0

        for ox, oy, h_total, spikes in self.obstacle_data:
            v_obs = np.array([ox, oy]) - cur_p[:2]
            proj = float(np.dot(v_obs, u_goal))

            if 0.15 < proj < min(d_goal, 3.5):
                lat_vec = v_obs - proj * u_goal
                lat_dist = float(np.linalg.norm(lat_vec))

                if lat_dist < r_corridor:
                    needed_clearance = r_corridor - lat_dist
                    deflect_angle = math.atan2(needed_clearance, proj)

                    cross = u_goal[0] * v_obs[1] - u_goal[1] * v_obs[0]
                    if abs(cross) < 1e-3:
                        steer_side = -1.0 if cur_p[0] > 0 else 1.0
                    else:
                        steer_side = -1.0 if cross > 0 else 1.0

                    if deflect_angle > max_deflect_angle:
                        max_deflect_angle = deflect_angle
                        best_deflect_dir = steer_side

        base_yaw = math.atan2(u_goal[1], u_goal[0])
        desired_yaw = base_yaw + best_deflect_dir * max_deflect_angle

        f_rep_xy = np.zeros(2, dtype=np.float32)
        for ox, oy, h_total, spikes in self.obstacle_data:
            dx = cur_p[0] - ox
            dy = cur_p[1] - oy
            d_center = math.hypot(dx, dy)
            d_surf = max(0.0, d_center - config.CYLINDER_RADIUS)
            norm_rep = np.array([dx / (d_center + 1e-6), dy / (d_center + 1e-6)])

            for (sp_x, sp_y, sp_z, sp_len, sp_rad) in spikes:
                tip_pos = np.array([ox + sp_x, oy + sp_y, sp_z])
                v_tip = cur_p - tip_pos
                dist_tip = float(np.linalg.norm(v_tip))
                d_sp = max(0.0, dist_tip - sp_rad)
                if d_sp < d_surf:
                    d_surf = d_sp
                    norm_rep = v_tip[:2] / (dist_tip + 1e-6)

            if d_surf < 0.6:
                rep_mag = (0.6 - d_surf) / 0.6
                f_rep_xy += rep_mag * norm_rep

        if np.linalg.norm(f_rep_xy) > 1e-3:
            v_des = np.array([math.cos(desired_yaw), math.sin(desired_yaw)]) + 1.2 * f_rep_xy
            desired_yaw = math.atan2(v_des[1], v_des[0])

        cur_yaw = float(cur_rpy[2])
        delta_yaw = (desired_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
        max_alpha_rad = math.radians(float(getattr(config, "MAX_ALPHA_DEG", 35.0)))
        alpha_safe_hint = float(np.clip(delta_yaw / max_alpha_rad, -1.0, 1.0))

        dz = goal_p[2] - cur_p[2]
        desired_pitch = math.atan2(dz, d_goal)
        max_beta_rad = math.radians(float(getattr(config, "MAX_BETA_DEG", 30.0)))
        beta_safe_hint = float(np.clip(desired_pitch / max_beta_rad, -1.0, 1.0))

        return alpha_safe_hint, beta_safe_hint

    def _get_obs(self):
        """Tạo không gian quan sát đa phương thức gồm ảnh FPV và toàn bộ vector trạng thái động học + đích + APF hints."""
        fpv_img = self._get_fpv_image()
        cur_p = self.pos[0]
        cur_v = self.vel[0]
        cur_rpy = self.rpy[0]
        g_p = self.goal[0]
        rel_vec = g_p - cur_p
        dist = float(np.linalg.norm(rel_vec))
        alpha_hint, beta_hint = self._compute_apf_hint()

        state_vec = np.array([
            float(cur_p[0]), float(cur_p[1]), float(cur_p[2]),
            float(cur_v[0]), float(cur_v[1]), float(cur_v[2]),
            float(cur_rpy[0]), float(cur_rpy[1]), float(cur_rpy[2]),
            float(g_p[0]), float(g_p[1]), float(g_p[2]),
            float(rel_vec[0]), float(rel_vec[1]), float(rel_vec[2]),
            dist,
            alpha_hint,
            beta_hint
        ], dtype=np.float32)

        return {
            "rgb": fpv_img,
            "state": state_vec
        }

    def _get_fpv_image(self):
        """Trích xuất ảnh quan sát FPV (64x64 RGB) từ camera gắn trên mũi drone."""
        rgb_img = self._get_drone_camera_image(drone_idx=0, camera_name="drone0_cam")
        return rgb_img

    def _get_drone_camera_image(self, drone_idx=0, camera_name="drone0_cam"):
        """Render ảnh camera offscreen thông qua MuJoCo Renderer (tái sử dụng bộ đệm renderer)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)

        if not hasattr(self, "_cam_renderer") or self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, height=config.IMG_HEIGHT, width=config.IMG_WIDTH)

        self._cam_renderer.update_scene(self.data, camera=cam_id)
        rgb = self._cam_renderer.render()
        return rgb

    def _generate_cactus_xml(self, start_pos, goal_pos, level):
        """Sinh chuỗi XML mô phỏng MuJoCo linh hoạt với các cột trụ có gai xương rồng ngẫu nhiên."""
        num_obstacles = config.OBSTACLE_COUNTS.get(level, 3)
        self.obstacle_data = []

        obstacle_bodies_xml = ""

        for i in range(num_obstacles):
            base_pos = config.OBSTACLE_POSITIONS[i]
            ox, oy = base_pos[0], base_pos[1]
            h_total = float(getattr(config, "FIXED_OBSTACLE_HEIGHT", 3.0))
            h_half = h_total / 2.0

            num_spikes = random.randint(config.SPIKES_PER_PILLAR_MIN, config.SPIKES_PER_PILLAR_MAX)
            spikes_list = []
            spikes_xml = ""

            for sp_idx in range(num_spikes):
                sp_z = float(np.random.uniform(0.3, h_total - 0.2))
                theta = float(np.random.uniform(0, 2 * math.pi))
                sp_rad = float(np.random.uniform(config.SPIKE_RADIUS_MIN, config.SPIKE_RADIUS_MAX))
                sp_len = float(np.random.uniform(config.SPIKE_LENGTH_MIN, config.SPIKE_LENGTH_MAX))

                dir_x = math.cos(theta)
                dir_y = math.sin(theta)
                dir_z = float(np.random.uniform(-0.1, 0.2))

                tip_rel_x = (config.CYLINDER_RADIUS + sp_len) * dir_x
                tip_rel_y = (config.CYLINDER_RADIUS + sp_len) * dir_y

                spikes_list.append((tip_rel_x, tip_rel_y, sp_z, sp_len, sp_rad))

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

        goal_x, goal_y = goal_pos[0], goal_pos[1]
        goal_radius = getattr(config, "GOAL_THRESHOLD", 2.0)
        goal_body_xml = f"""
    <body name="goal_pole" pos="{goal_x} {goal_y} 1.5">
      <geom type="cylinder" size="0.15 1.5" rgba="1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 1.6" size="0.4" rgba="1 0.2 0.2 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 -1.45" size="{goal_radius} 0.02" rgba="1 0.2 0.2 0.25" contype="0" conaffinity="0"/>
    </body>"""

        y_min_limit = float(goal_pos[1] - config.OVERMAP_PAST_GOAL_DISTANCE)
        y_max_limit = float(start_pos[1] + config.OVERMAP_BEHIND_START_DISTANCE)

        obs_xs = [obs[0] for obs in self.obstacle_data]
        min_obs_x = min(obs_xs) if obs_xs else -5.0
        max_obs_x = max(obs_xs) if obs_xs else 5.0
        x_min_limit = float(min(min_obs_x - config.OVERMAP_X_MARGIN, start_pos[0] - 0.5, goal_pos[0] - 0.5))
        x_max_limit = float(max(max_obs_x + config.OVERMAP_X_MARGIN, start_pos[0] + 0.5, goal_pos[0] + 0.5))

        self.x_min_limit = x_min_limit
        self.x_max_limit = x_max_limit
        self.y_min_limit = y_min_limit
        self.y_max_limit = y_max_limit

        len_x = abs(x_max_limit - x_min_limit)
        len_y = abs(y_max_limit - y_min_limit)
        mid_x = (x_min_limit + x_max_limit) / 2.0
        mid_y = (y_min_limit + y_max_limit) / 2.0
        line_w = 0.035
        line_h = 0.005

        boundary_lines_xml = f"""
    <!-- 4 Đường viền trắng chạy trên mặt đất giới hạn phạm vi bay của Drone cho Level {level} -->
    <geom name="boundary_line_left" type="box" pos="{x_min_limit:.3f} {mid_y:.3f} 0.006" size="{line_w} {len_y / 2.0:.3f} {line_h}" rgba="1 1 1 0.95" contype="0" conaffinity="0"/>
    <geom name="boundary_line_right" type="box" pos="{x_max_limit:.3f} {mid_y:.3f} 0.006" size="{line_w} {len_y / 2.0:.3f} {line_h}" rgba="1 1 1 0.95" contype="0" conaffinity="0"/>
    <geom name="boundary_line_back" type="box" pos="{mid_x:.3f} {y_max_limit:.3f} 0.006" size="{len_x / 2.0:.3f} {line_w} {line_h}" rgba="1 1 1 0.95" contype="0" conaffinity="0"/>
    <geom name="boundary_line_front" type="box" pos="{mid_x:.3f} {y_min_limit:.3f} 0.006" size="{len_x / 2.0:.3f} {line_w} {line_h}" rgba="1 1 1 0.95" contype="0" conaffinity="0"/>"""

        drone_x, drone_y, drone_z = start_pos[0], start_pos[1], start_pos[2]
        show_m = getattr(self, "show_marker", False)
        marker_xml = ""
        if show_m:
            marker_xml = """
      <geom name="drone0_arrow_stem" type="cylinder" pos="0 0 1.8" size="0.12 0.7" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom name="drone0_arrow_pointer" type="sphere" pos="0 0 0.8" size="0.4" rgba="1 1 1 1" contype="0" conaffinity="0"/>"""

        meshdir = str(CF2_MESH_DIR).replace("\\", "/")
        visual_meshes = "\n".join(
            f'    <mesh file="{meshdir}/cf2_{i}.obj" name="cf2_vis_{i}"/>'
            for i in range(7)
        )
        cf2_materials = """    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>"""

        drone_xml = f"""
    <body name="drone0" pos="{drone_x} {drone_y} {drone_z}" quat="0.707 0 0 -0.707">
      <freejoint name="drone0_joint"/>
      <inertial pos="0 0 0" mass="0.027" diaginertia="1.4e-5 1.4e-5 2.1e-5"/>
      <geom name="drone0_collision" type="cylinder" size="0.06 0.025" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom mesh="cf2_vis_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_vis_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_vis_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_vis_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_vis_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_vis_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_vis_6" material="white" class="visual"/>{marker_xml}
      <site name="drone0_center" pos="0 0 0" group="5"/>
      <site name="drone0_prop0" pos="0.028 0.028 0" group="5"/>
      <site name="drone0_prop1" pos="-0.028 0.028 0" group="5"/>
      <site name="drone0_prop2" pos="-0.028 -0.028 0" group="5"/>
      <site name="drone0_prop3" pos="0.028 -0.028 0" group="5"/>
      <camera name="drone0_cam" pos="0.03 0 0.01" xyaxes="0 -1 0 0 0 1" fovy="{getattr(config, 'CAMERA_FOVY', 75.0)}"/>
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
    <quality shadowsize="0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{cf2_materials}
{visual_meshes}
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="20 30 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_xml}
{obstacle_bodies_xml}
{goal_body_xml}
{boundary_lines_xml}
  </worldbody>

  <sensor>
{sensors_xml}
  </sensor>
</mujoco>"""
        return full_xml

    def _apply_camera_settings(self):
        """Áp dụng góc nhìn camera (ưu tiên góc người dùng đã chỉnh, hoặc mặc định từ config)."""
        if self._viewer is None or not hasattr(self._viewer, "cam"):
            return

        cam = self._viewer.cam
        is_tracking = getattr(config, "GUI_CAMERA_TRACKING", True)

        if is_tracking:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
            if body_id >= 0:
                cam.trackbodyid = body_id
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if self.has_user_camera:
                cam.lookat[:] = self.saved_cam_lookat
            else:
                cam.lookat[:] = [float(self.pos[0, 0]), float(self.pos[0, 1]), float(self.pos[0, 2])]

        cam.distance = float(self.saved_cam_distance)
        cam.elevation = float(self.saved_cam_elevation)
        cam.azimuth = float(self.saved_cam_azimuth)

    def _capture_camera_settings(self):
        """Ghi nhớ lại các góc nhìn, khoảng cách người dùng vừa lướt chuột căn chỉnh."""
        if self._viewer is not None and hasattr(self._viewer, "cam") and self._viewer.is_running():
            cam = self._viewer.cam
            self.saved_cam_azimuth = float(cam.azimuth)
            self.saved_cam_elevation = float(cam.elevation)
            self.saved_cam_distance = float(cam.distance)
            self.saved_cam_lookat = np.array(cam.lookat, dtype=float).copy()
            self.has_user_camera = True

    def render(self, camera_mode=None, track_drone_id=0):
        """Render giao diện 3D với 2 cửa sổ song song:
        1. Cửa sổ MuJoCo 3D Viewer: Góc nhìn thứ 3 toàn cảnh (Overview / Tracking).
        2. Cửa sổ Tkinter FPV Window: Góc nhìn thứ nhất (POV trực tiếp từ mũi Drone).
        """
        if self.gui_mode or self.render_mode == "human":
            if self._viewer is None:
                show_r = getattr(config, "GUI_SHOW_RIGHT_UI", False)
                show_l = getattr(config, "GUI_SHOW_LEFT_UI", False)
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=show_l, show_right_ui=show_r
                )
                self._apply_camera_settings()

            self._viewer.sync()
            self._capture_camera_settings()

            if getattr(config, "GUI_SHOW_POV_WINDOW", True):
                self._render_fpv_window()

        elif self.render_mode == "rgb_array":
            return super().render(camera_mode=camera_mode, track_drone_id=track_drone_id)

    def _render_fpv_window(self):
        """Hiển thị góc nhìn POV (Camera FPV mũi Drone) trên cửa sổ riêng biệt bằng Tkinter."""
        try:
            import tkinter as tk
            from PIL import Image, ImageTk

            fpv_img = self._get_fpv_image()
            win_size = getattr(config, "GUI_POV_WINDOW_SIZE", 256)

            if self._fpv_window is None:
                self._fpv_window = tk.Tk()
                self._fpv_window.title("Drone POV (Góc nhìn FPV mũi Drone)")
                self._fpv_window.geometry(f"{win_size}x{win_size}+50+50")
                self._fpv_window.resizable(False, False)

                self._fpv_label = tk.Label(self._fpv_window, bg="black")
                self._fpv_label.pack(fill="both", expand=True)

                def on_close():
                    if self._fpv_window is not None:
                        try:
                            self._fpv_window.destroy()
                        except Exception:
                            pass
                        self._fpv_window = None

                self._fpv_window.protocol("WM_DELETE_WINDOW", on_close)

            if self._fpv_window is not None and self._fpv_label is not None:
                img_pil = Image.fromarray(fpv_img).resize((win_size, win_size), Image.Resampling.NEAREST)
                self._fpv_img_tk = ImageTk.PhotoImage(img_pil)
                self._fpv_label.config(image=self._fpv_img_tk)

                self._fpv_window.update_idletasks()
                self._fpv_window.update()

        except Exception:
            self._fpv_window = None

    def close(self):
        """Đóng môi trường và giải phóng an toàn cả 2 cửa sổ."""
        if hasattr(self, "_fpv_window") and self._fpv_window is not None:
            try:
                self._fpv_window.destroy()
            except Exception:
                pass
            self._fpv_window = None

        if hasattr(self, "_viewer") and self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

        if hasattr(self, "_cam_renderer") and self._cam_renderer is not None:
            try:
                self._cam_renderer.close()
            except Exception:
                pass
            self._cam_renderer = None

        super().close()

