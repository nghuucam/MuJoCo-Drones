import numpy as np

start_space = [
    np.array([1, 11]),
    np.array([-2, 11]),
    np.array([2, 11]),
    np.array([-3, 11]),
    np.array([3, 11]),
    np.array([-4, 11]),
    np.array([4, 11])
]

goal_space = [
    np.array([2, -10]),
    np.array([1, -11]),
    np.array([-2, -11]),
    np.array([-1, -10]),
    np.array([1, -10]),
    np.array([2, -11]),
    np.array([-2, -10])
]

obstacle_position = [
    [0.0, 9.0],
    [-2.0, 7.5],
    [2.0, 7.5],
    [-1.0, 5.0],
    [1.5, 5.0],
    [0.0, 2.5],
    [-2.5, 1.0],
    [2.5, 1.0],
    [-1.0, -2.5],
    [1.0, -2.5],
    [0.0, -5.0],
    [-2.0, -7.0],
    [2.0, -7.0],
    [-0.5, -9.0],
    [1.5, -9.0]
]
