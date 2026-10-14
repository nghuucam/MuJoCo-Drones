NUM_ENVS = 128
SIM_FREQ = 100
SUBSTEPS = 50
MAX_STEPS = 100
ACTION_DURATION = SUBSTEPS / SIM_FREQ

START_POS = (0.0, 11.0, 1.0)
START_YAW = -1.5707963
GOAL_Z = 1.5

MAX_ALPHA_RAD = 0.610865
MAX_BETA_RAD = 0.523599
MIN_STEP_DIST = 0.2
STEP_DIST_SCALE = 0.65

TRACK_KP = 200.0
TRACK_KD = 28.0
MAX_FORCE = 2.0

GOAL_THRESHOLD = 0.5
CYLINDER_RADIUS = 0.50
CYLINDER_HEIGHT = 3.00

MIN_Z = 0.25
MAX_Z = 2.50
X_LIMIT = 6.50
Y_LIMIT = 13.00

GOAL_Y_RANGES = [
    (7.5, 8.5),
    (-0.5, 0.5),
    (-0.5, 0.5),
    (-5.5, -4.5),
    (-5.5, -4.5),
    (-10.5, -9.5),
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
