import numpy as np

NUM_ENVS = 128
SIM_FREQ = 200
SUBSTEPS = 50
MAX_STEPS = 100

GOAL_THRESHOLD = 0.5
COLLISION_MARGIN = 0.05
CYLINDER_RADIUS = 0.50
CYLINDER_HEIGHT = 3.00

MIN_Z = 0.25
MAX_Z = 2.50
X_LIMIT = 6.50
Y_LIMIT = 13.00

GOAL_Y_RANGES = [
    (7.5, 8.5),
    (3.5, 4.5),
    (-0.5, 0.5),
    (-3.5, -2.5),
    (-5.5, -4.5),
    (-8.5, -7.5),
    (-10.5, -9.5)
]

OBSTACLE_COUNTS = [0, 3, 6, 8, 10, 12, 15]

FIXED_OBSTACLE_POSITIONS = [
    [-4.50,  7.50],
    [ 0.00,  7.00],
    [ 4.50,  7.50],
    [-2.50,  4.00],
    [ 2.50,  4.00],
    [ 0.00,  2.00],
    [-4.00,  0.50],
    [ 4.00,  0.50],
    [-2.20, -2.50],
    [ 2.20, -2.50],
    [-4.20, -5.00],
    [ 0.00, -5.50],
    [ 4.20, -5.00],
    [-2.00, -8.00],
    [ 2.00, -8.00]
]

CF2_MASS = 0.027
GRAVITY = 9.81
KF = 3.16e-10
KM = 7.94e-12
ARM_LENGTH = 0.0397
MAX_RPM = 21714.0
HOVER_RPM = np.sqrt((CF2_MASS * GRAVITY) / (4 * KF))
