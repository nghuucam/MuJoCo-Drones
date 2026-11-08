"""Curriculum learning wrapper for progressive difficulty scaling.

Automatically adjusts task difficulty based on agent performance,
enabling smooth learning from easy to hard tasks.
"""

import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
from collections import deque


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Parameters
    ----------
    metric : str
        Performance metric to track ("success_rate", "reward", "episode_length").
    threshold_advance : float
        Metric value above which difficulty increases.
    threshold_retreat : float
        Metric value below which difficulty decreases.
    window_size : int
        Number of episodes to average metric over.
    num_levels : int
        Total number of difficulty levels.
    start_level : int
        Initial difficulty level (0 = easiest).
    advance_count : int
        Must exceed threshold for this many consecutive windows to advance.
    """
    metric: str = "success_rate"
    threshold_advance: float = 0.8
    threshold_retreat: float = 0.2
    window_size: int = 50
    num_levels: int = 10
    start_level: int = 0
    advance_count: int = 1


class CurriculumWrapper(gym.Wrapper):
    """Gymnasium wrapper that implements automatic curriculum learning.

    The wrapper tracks agent performance and adjusts environment difficulty
    by calling a user-provided `difficulty_fn(env, level)` on each reset.

    Example
    -------
    >>> def adjust_difficulty(env, level):
    ...     # Level 0-9: target gets farther
    ...     env.TARGET_HEIGHT = 0.3 + level * 0.1
    ...     # Level 5+: add wind
    ...     if level >= 5:
    ...         env.wind_speed = (level - 5) * 0.5
    ...
    >>> env = CurriculumWrapper(HoverAviary(), difficulty_fn=adjust_difficulty)
    """

    def __init__(
        self,
        env: gym.Env,
        difficulty_fn: Callable[[gym.Env, int], None],
        config: Optional[CurriculumConfig] = None,
    ):
        super().__init__(env)
        self.config = config or CurriculumConfig()
        self.difficulty_fn = difficulty_fn
        self.current_level = self.config.start_level

        # Tracking
        self._episode_metrics: deque = deque(maxlen=self.config.window_size)
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False
        self._advance_streak = 0

    @property
    def level(self) -> int:
        return self.current_level

    @property
    def progress(self) -> float:
        """Normalized progress through curriculum [0, 1]."""
        return self.current_level / max(self.config.num_levels - 1, 1)

    def reset(self, **kwargs):
        # Apply difficulty for current level
        self.difficulty_fn(self.env, self.current_level)

        # Reset episode tracking
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False

        obs, info = self.env.reset(**kwargs)
        info["curriculum_level"] = self.current_level
        info["curriculum_progress"] = self.progress
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward
        self._episode_steps += 1

        # Check for success flag in info
        if info.get("success", False) or info.get("is_success", False):
            self._episode_success = True

        # On episode end, update curriculum
        if terminated or truncated:
            self._record_episode(info)
            self._maybe_adjust_level()
            info["curriculum_level"] = self.current_level

        return obs, reward, terminated, truncated, info

    def _record_episode(self, info: Dict[str, Any]):
        """Record episode metrics."""
        metric = self.config.metric
        if metric == "success_rate":
            value = float(self._episode_success)
        elif metric == "reward":
            value = self._episode_reward
        elif metric == "episode_length":
            value = self._episode_steps
        else:
            value = info.get(metric, 0.0)
        self._episode_metrics.append(value)

    def _maybe_adjust_level(self):
        """Check if we should advance or retreat."""
        if len(self._episode_metrics) < self.config.window_size:
            return

        avg = np.mean(list(self._episode_metrics))

        if avg >= self.config.threshold_advance:
            self._advance_streak += 1
            if self._advance_streak >= self.config.advance_count:
                self.current_level = min(self.current_level + 1, self.config.num_levels - 1)
                self._advance_streak = 0
                self._episode_metrics.clear()
        elif avg <= self.config.threshold_retreat:
            self.current_level = max(self.current_level - 1, 0)
            self._advance_streak = 0
            self._episode_metrics.clear()
        else:
            self._advance_streak = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return curriculum statistics."""
        metrics = list(self._episode_metrics)
        return {
            "level": self.current_level,
            "progress": self.progress,
            "metric_avg": np.mean(metrics) if metrics else 0.0,
            "metric_std": np.std(metrics) if metrics else 0.0,
            "episodes_tracked": len(metrics),
        }
