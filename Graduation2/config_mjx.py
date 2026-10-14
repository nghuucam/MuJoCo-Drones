import numpy as np

NUM_ENVS = 1024
SIM_FREQ = 240
CTRL_FREQ = 48
SUBSTEPS = SIM_FREQ // CTRL_FREQ
MAX_STEPS = 140

GOAL_THRESHOLD = 0.5
COLLISION_MARGIN = 0.05
CYLINDER_RADIUS = 0.30
CYLINDER_HEIGHT = 2.50

MIN_Z = 0.8
MAX_Z = 2.5
X_LIMIT = 7.0
Y_LIMIT = 12.5

GOAL_Y_RANGES = [
    (-6.0, -5.0),
    (-8.5, -7.0),
    (-11.0, -9.5),
    (-13.5, -12.0),
    (-16.0, -14.5),
    (-18.5, -17.0),
    (-21.0, -19.0)
]

OBSTACLE_COUNTS = [0, 2, 4, 7, 10, 12, 15]

FIXED_OBSTACLE_POSITIONS = [
    [-1.5,  5.0],
    [ 1.5,  5.0],
    [ 0.0,  2.5],
    [-2.0,  0.0],
    [ 2.0,  0.0],
    [-1.0, -2.5],
    [ 1.0, -2.5],
    [ 0.0, -5.0],
    [-2.0, -7.5],
    [ 2.0, -7.5],
    [-1.0,-10.0],
    [ 1.0,-10.0],
    [ 0.0,-12.5],
    [-2.0,-15.0],
    [ 2.0,-15.0]
]

CF2_MASS = 0.027
GRAVITY = 9.81
KF = 3.16e-10
KM = 7.94e-12
ARM_LENGTH = 0.0397
MAX_RPM = 21714.0
HOVER_RPM = np.sqrt((CF2_MASS * GRAVITY) / (4 * KF))
