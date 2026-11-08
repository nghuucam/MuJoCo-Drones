"""gym-mujoco-drones: Multi-drone environments for MuJoCo.

A gymnasium-compatible multi-drone simulation environment using MuJoCo physics,
inspired by gym-pybullet-drones but with superior performance, accuracy, and features.

Uses the Bitcraze Crazyflie 2.x model from mujoco_menagerie.
"""

__version__ = "1.0.0"

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary
from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
from multi_drone_mujoco.envs.formation_aviary import FormationAviary
from multi_drone_mujoco.envs.race_aviary import RaceAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
