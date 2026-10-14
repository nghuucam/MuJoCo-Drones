import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MJXCurriculumCallback(BaseCallback):
    def __init__(
        self,
        window_size: int = 100,
        threshold_advance: float = 0.8,
        threshold_retreat: float = 0.2,
        num_levels: int = 7,
        start_level: int = 0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.threshold_advance = threshold_advance
        self.threshold_retreat = threshold_retreat
        self.num_levels = num_levels
        self.current_level = start_level
        self.episode_outcomes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                if "terminal_observation" in info or info.get("win") or info.get("collision") or info.get("over_map"):
                    is_win = bool(info.get("win", False))
                    self.episode_outcomes.append(1 if is_win else 0)

        if len(self.episode_outcomes) >= self.window_size:
            recent_outcomes = self.episode_outcomes[-self.window_size:]
            win_rate = float(np.mean(recent_outcomes))

            if hasattr(self, "logger") and self.logger is not None:
                self.logger.record("curriculum/win_rate", win_rate)
                self.logger.record("curriculum/level", self.current_level)

            if win_rate >= self.threshold_advance and self.current_level < self.num_levels - 1:
                old_lvl = self.current_level
                self.current_level += 1
                self.episode_outcomes.clear()
                self._update_env_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n{'='*75}")
                    print(f"🎉 [CURRICULUM THĂNG CẤP] Win Rate: {win_rate*100:.1f}% >= {self.threshold_advance*100:.0f}%")
                    print(f"🚀 Chuyển từ Level {old_lvl} ➔ Level {self.current_level}!")
                    print(f"{'='*75}\n")

            elif win_rate <= self.threshold_retreat and self.current_level > 0:
                old_lvl = self.current_level
                self.current_level -= 1
                self.episode_outcomes.clear()
                self._update_env_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n{'='*75}")
                    print(f"⚠️ [CURRICULUM HẠ CẤP] Win Rate: {win_rate*100:.1f}% <= {self.threshold_retreat*100:.0f}%")
                    print(f"🔻 Lùi từ Level {old_lvl} ➔ Level {self.current_level} để củng cố kỹ năng!")
                    print(f"{'='*75}\n")

        return True

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
