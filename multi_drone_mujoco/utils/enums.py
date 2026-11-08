"""Enumerations for the multi-drone MuJoCo environment."""

from enum import Enum


class DroneModel(Enum):
    """Drone models available."""
    CF2X = "cf2x"       # Crazyflie 2.x in X configuration
    CF2P = "cf2p"       # Crazyflie 2.x in + configuration
    RACE = "race"       # Racing drone (larger, more powerful)


class Physics(Enum):
    """Physics implementations available."""
    MJC = "mjc"                         # Pure MuJoCo physics
    DYN = "dyn"                         # Explicit dynamics (no MuJoCo stepping)
    MJC_GND = "mjc_gnd"                 # MuJoCo + ground effect
    MJC_DRAG = "mjc_drag"               # MuJoCo + drag
    MJC_DW = "mjc_dw"                   # MuJoCo + downwash
    MJC_GND_DRAG_DW = "mjc_gnd_drag_dw" # MuJoCo + all effects


class ActionType(Enum):
    """Action types for the environment."""
    RPM = "rpm"          # Direct RPM control of 4 motors
    VEL = "vel"          # Velocity vector [vx, vy, vz, yaw_rate]
    ONE_D_RPM = "one_d_rpm"  # 1D thrust (same RPM to all motors)
    PID = "pid"          # Target position [x, y, z, yaw] -> internal PID
    ATTITUDE = "attitude"  # Thrust + roll + pitch + yaw_rate


class ObservationType(Enum):
    """Observation types for the environment."""
    KIN = "kin"          # Kinematics only (pos, quat, rpy, vel, angvel)
    RGB = "rgb"          # RGB camera image
    KIN_RGB = "kin_rgb"  # Both kinematics and RGB


class ImageType(Enum):
    """Image types for rendering."""
    RGB = "rgb"
    DEP = "dep"
    SEG = "seg"
    BW = "bw"
