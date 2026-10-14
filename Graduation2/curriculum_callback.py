from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import config_mjx as cfg


class MJXCurriculumCallback(BaseCallback):
    def __init__(
        self,
        window_size: int = 100,
        threshold_advance: float = 0.8,
        threshold_retreat: float = 0.2,
        num_levels: int = len(cfg.GOAL_Y_RANGES),
        start_level: int = 0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.threshold_advance = threshold_advance
        self.threshold_retreat = threshold_retreat
        self.num_levels = num_levels
        self.current_level = start_level
        self.episode_outcomes = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("curriculum/level", float(self.current_level))

        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "is_success" in info:
                if "terminal_observation" in info or info.get("collision") or info.get("win") or info.get("over_map"):
                    is_succ = float(info.get("is_success", False))
                    self.episode_outcomes.append(is_succ)
                    self._maybe_adjust_level()

        return True

    def _maybe_adjust_level(self):
        if len(self.episode_outcomes) < self.window_size:
            return

        success_rate = float(np.mean(list(self.episode_outcomes)))
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("curriculum/success_rate", success_rate)
            self.logger.record("curriculum/win_rate", success_rate)

        if success_rate >= self.threshold_advance:
            if self.current_level < self.num_levels - 1:
                old_lvl = self.current_level
                self.current_level += 1
                self.episode_outcomes.clear()
                self._update_env_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n{'='*75}")
                    print(f"🎉 [CURRICULUM UP] Success Rate: {success_rate*100:.1f}% >= {self.threshold_advance*100:.0f}%")
                    print(f"🚀 TỰ ĐỘNG CHUYỂN SANG CURRICULUM LEVEL {self.current_level}!")
                    print(f"{'='*75}\n")

        elif success_rate <= self.threshold_retreat:
            if self.current_level > 0:
                old_lvl = self.current_level
                self.current_level -= 1
                self.episode_outcomes.clear()
                self._update_env_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n{'='*75}")
                    print(f"⚠️ [CURRICULUM DOWN] Success Rate: {success_rate*100:.1f}% <= {self.threshold_retreat*100:.0f}%")
                    print(f"📉 HẠ BỚT ĐỘ KHÓ VỀ LEVEL {self.current_level} để drone học lại nền tảng!")
                    print(f"{'='*75}\n")

    def _init_callback(self) -> None:
        self._update_env_level(self.current_level)

    def _update_env_level(self, new_level: int):
        training_env = self.training_env
        if hasattr(training_env, "envs"):
            for env in training_env.envs:
                if hasattr(env, "set_level"):
                    env.set_level(new_level)
        elif hasattr(training_env, "set_level"):
            training_env.set_level(new_level)
        elif hasattr(training_env, "env_method"):
            training_env.env_method("set_level", new_level)
