import os
import csv
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class FlightLoggerCallback(BaseCallback):
    """Callback tự động ghi lại nhật ký chuyến bay (Excel-compatible CSV) tương tự D3QN.
    
    Hỗ trợ cả huấn luyện đơn tiến trình (train_ppo_curriculum.py) và song song (train_parallel_curriculum.py).
    Đảm bảo:
    1. 100% tương thích với cấu trúc phân tích của ana1.py (Eposide, AccumReward, Win, Collision...).
    2. Gom các bước bay theo từng episode liền mạch, không bị đan xen dòng giữa các worker CPU.
    3. Tự động flush ra đĩa sau mỗi episode để có thể mở file xem ngay trong Excel.
    4. Kháng lỗi khoá file (PermissionError) khi người dùng đang mở file xem trong Excel.
    """
    def __init__(self, log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.csv_writer = None
        self.global_episode_id = 0
        
        # Buffer theo dõi từng worker: worker_idx -> list of rows
        self.env_buffers = {}
        self.env_accum_rewards = {}

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        file_exists = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        
        try:
            self.log_file = open(self.log_path, mode='a', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.log_file)
            
            if not file_exists:
                # Header kế thừa 100% tên cột từ D3QN + bổ sung Level và Reward từng bước
                self.csv_writer.writerow([
                    'Eposide',
                    'Step',
                    'Level',
                    'Start',
                    'End',
                    'X',
                    'Y',
                    'Z',
                    'Collision',
                    'Win',
                    'Over_Step',
                    'Over_Map',
                    'Reward',
                    'AccumReward',
                    'Action'
                ])
                self.log_file.flush()
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ [FlightLoggerCallback] Không thể mở file log: {e}")

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        
        n_envs = len(infos)
        for i in range(n_envs):
            if i not in self.env_buffers:
                self.env_buffers[i] = []
                self.env_accum_rewards[i] = 0.0

            info = infos[i]
            r = float(rewards[i])
            self.env_accum_rewards[i] += r
            
            # Đếm số bước của episode
            step = info.get("step_count", len(self.env_buffers[i]) + 1)
            level = info.get("curriculum_level", 0)
            
            start = info.get("start_pos", [0.0, 11.0, 1.0])
            start_str = f"[{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}]"
            
            end = info.get("goal_pos", [0.0, 0.0, 1.5])
            end_str = f"[{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}]"
            
            pos = info.get("drone_pos", [0.0, 0.0, 0.0])
            x, y, z = round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)
            
            col = 1 if info.get("collision") else 0
            win = 1 if info.get("win") else 0
            over_step = 1 if info.get("over_step") else 0
            over_map = 1 if info.get("over_map") else 0
            
            # Action: [alpha, beta, d]
            if actions is not None and len(actions) > i:
                act = actions[i]
                if hasattr(act, "tolist"):
                    act_list = act.tolist()
                else:
                    act_list = list(act)
                action_str = f"[{act_list[0]:.2f}, {act_list[1]:.2f}, {act_list[2]:.2f}]"
            else:
                action_str = "[]"
            
            row = [
                None, # Sẽ gán ID Episode khi hoàn thành
                step,
                level,
                start_str,
                end_str,
                x,
                y,
                z,
                col,
                win,
                over_step,
                over_map,
                round(r, 3),
                round(self.env_accum_rewards[i], 3),
                action_str
            ]
            self.env_buffers[i].append(row)
            
            # Khi episode kết thúc (done == True)
            if dones is not None and dones[i]:
                self.global_episode_id += 1
                
                # Ghi toàn bộ chuỗi bước của episode này vào CSV
                if self.csv_writer is not None:
                    try:
                        for r_data in self.env_buffers[i]:
                            r_data[0] = self.global_episode_id
                            self.csv_writer.writerow(r_data)
                        self.log_file.flush()
                    except PermissionError:
                        # Tránh crash nếu người dùng đang tạm thời khóa file trong Excel
                        pass
                
                # Reset buffer cho worker này
                self.env_buffers[i] = []
                self.env_accum_rewards[i] = 0.0

        return True

    def _on_training_end(self) -> None:
        if self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass
