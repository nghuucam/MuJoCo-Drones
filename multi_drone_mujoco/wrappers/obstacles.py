"""Procedural obstacle generation for drone environments.

Generates randomized obstacle-rich environments:
- Forest (cylinder trees)
- Urban canyon (box buildings)
- Indoor room (walls + furniture)
- Random clutter
- Custom from config

Obstacles are added as MuJoCo geoms in the world body XML.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ObstacleType(Enum):
    """Pre-defined obstacle environment types."""
    NONE = "none"
    FOREST = "forest"
    URBAN = "urban"
    INDOOR = "indoor"
    RANDOM = "random"
    GATES = "gates"          # Racing gates
    CUSTOM = "custom"


@dataclass
class Obstacle:
    """Single obstacle definition."""
    geom_type: str           # "box", "cylinder", "sphere", "capsule"
    position: np.ndarray     # [x, y, z]
    size: np.ndarray         # depends on type (half-sizes for box, radius+half-height for cylinder)
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ObstacleConfig:
    """Configuration for procedural obstacle generation.

    Parameters
    ----------
    obstacle_type : ObstacleType
        Pre-defined environment layout.
    num_obstacles : int
        Number of obstacles to generate.
    arena_size : tuple
        (x_half, y_half, z_max) for the arena bounds.
    min_spacing : float
        Minimum distance between obstacle centers.
    seed : int or None
        Random seed for reproducibility.
    safe_zone_radius : float
        No obstacles within this radius of spawn points.
    custom_obstacles : list
        List of Obstacle objects for CUSTOM type.
    """
    obstacle_type: ObstacleType = ObstacleType.NONE
    num_obstacles: int = 20
    arena_size: tuple = (3.0, 3.0, 2.5)
    min_spacing: float = 0.3
    seed: Optional[int] = None
    safe_zone_radius: float = 0.5
    safe_zone_centers: Optional[np.ndarray] = None  # (N, 3) spawn positions
    custom_obstacles: List[Obstacle] = field(default_factory=list)


def generate_obstacles(config: ObstacleConfig) -> List[Obstacle]:
    """Generate obstacle list from config.

    Returns
    -------
    obstacles : list of Obstacle
        Generated obstacles.
    """
    if config.obstacle_type == ObstacleType.NONE:
        return []
    if config.obstacle_type == ObstacleType.CUSTOM:
        return config.custom_obstacles

    rng = np.random.default_rng(config.seed)
    generators = {
        ObstacleType.FOREST: _generate_forest,
        ObstacleType.URBAN: _generate_urban,
        ObstacleType.INDOOR: _generate_indoor,
        ObstacleType.RANDOM: _generate_random,
        ObstacleType.GATES: _generate_gates,
    }
    return generators[config.obstacle_type](config, rng)


def obstacles_to_xml(obstacles: List[Obstacle]) -> str:
    """Convert obstacle list to MuJoCo XML body elements.

    Returns
    -------
    xml : str
        XML string to insert into worldbody.
    """
    if not obstacles:
        return ""
    lines = []
    for i, obs in enumerate(obstacles):
        pos_str = f"{obs.position[0]:.4f} {obs.position[1]:.4f} {obs.position[2]:.4f}"
        rgba_str = f"{obs.rgba[0]:.3f} {obs.rgba[1]:.3f} {obs.rgba[2]:.3f} {obs.rgba[3]:.3f}"

        if obs.geom_type == "box":
            size_str = f"{obs.size[0]:.4f} {obs.size[1]:.4f} {obs.size[2]:.4f}"
        elif obs.geom_type in ("cylinder", "capsule"):
            size_str = f"{obs.size[0]:.4f} {obs.size[1]:.4f}"
        elif obs.geom_type == "sphere":
            size_str = f"{obs.size[0]:.4f}"
        else:
            size_str = " ".join(f"{s:.4f}" for s in obs.size)

        euler_str = f"{obs.euler[0]:.3f} {obs.euler[1]:.3f} {obs.euler[2]:.3f}"
        lines.append(
            f'    <body name="obstacle_{i}" pos="{pos_str}">'
            f'\n      <geom type="{obs.geom_type}" size="{size_str}" '
            f'rgba="{rgba_str}" euler="{euler_str}" '
            f'contype="1" conaffinity="1"/>'
            f'\n    </body>'
        )
    return "\n".join(lines)


def _try_place(position: np.ndarray, placed: List[np.ndarray],
               config: ObstacleConfig) -> bool:
    """Check if position is valid (min spacing + safe zone)."""
    # Check safe zones
    if config.safe_zone_centers is not None:
        for center in config.safe_zone_centers:
            if np.linalg.norm(position[:2] - center[:2]) < config.safe_zone_radius:
                return False
    # Check existing obstacles
    for p in placed:
        if np.linalg.norm(position[:2] - p[:2]) < config.min_spacing:
            return False
    return True


def _generate_forest(config: ObstacleConfig, rng: np.random.Generator) -> List[Obstacle]:
    """Generate forest of cylindrical trees."""
    obstacles = []
    placed = []
    xh, yh, zh = config.arena_size
    attempts = 0

    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        radius = rng.uniform(0.03, 0.12)
        height = rng.uniform(0.5, zh)
        pos = np.array([x, y, height / 2])

        if not _try_place(pos, placed, config):
            continue

        # Brown/green tree colors
        g = rng.uniform(0.2, 0.5)
        rgba = np.array([rng.uniform(0.3, 0.6), g, rng.uniform(0.1, 0.3), 1.0])
        obstacles.append(Obstacle(
            geom_type="cylinder",
            position=pos,
            size=np.array([radius, height / 2]),
            rgba=rgba,
        ))
        placed.append(pos)

    return obstacles


def _generate_urban(config: ObstacleConfig, rng: np.random.Generator) -> List[Obstacle]:
    """Generate urban canyon with box buildings."""
    obstacles = []
    placed = []
    xh, yh, zh = config.arena_size
    attempts = 0

    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        sx = rng.uniform(0.1, 0.5)
        sy = rng.uniform(0.1, 0.5)
        sz = rng.uniform(0.3, zh)
        pos = np.array([x, y, sz / 2])

        if not _try_place(pos, placed, config):
            continue

        # Gray concrete colors
        gray = rng.uniform(0.3, 0.7)
        rgba = np.array([gray, gray, gray * rng.uniform(0.9, 1.1), 1.0])
        obstacles.append(Obstacle(
            geom_type="box",
            position=pos,
            size=np.array([sx, sy, sz / 2]),
            rgba=rgba,
        ))
        placed.append(pos)

    return obstacles


def _generate_indoor(config: ObstacleConfig, rng: np.random.Generator) -> List[Obstacle]:
    """Generate indoor room with walls and furniture-like obstacles."""
    obstacles = []
    xh, yh, zh = config.arena_size

    # Walls (thin boxes)
    wall_thickness = 0.05
    wall_height = zh
    walls = [
        # Four walls
        Obstacle("box", np.array([xh, 0, wall_height/2]), np.array([wall_thickness, yh, wall_height/2]), np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([-xh, 0, wall_height/2]), np.array([wall_thickness, yh, wall_height/2]), np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([0, yh, wall_height/2]), np.array([xh, wall_thickness, wall_height/2]), np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([0, -yh, wall_height/2]), np.array([xh, wall_thickness, wall_height/2]), np.array([0.8, 0.8, 0.75, 1.0])),
        # Ceiling
        Obstacle("box", np.array([0, 0, zh]), np.array([xh, yh, wall_thickness]), np.array([0.9, 0.9, 0.85, 1.0])),
    ]
    obstacles.extend(walls)

    # Interior obstacles (shelves, tables, columns)
    placed = [w.position for w in walls]
    n_interior = config.num_obstacles - len(walls)
    attempts = 0
    while len(obstacles) - len(walls) < n_interior and attempts < n_interior * 20:
        attempts += 1
        x = rng.uniform(-xh * 0.8, xh * 0.8)
        y = rng.uniform(-yh * 0.8, yh * 0.8)
        kind = rng.choice(["table", "column", "shelf"])

        if kind == "table":
            sx, sy, sz = rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5), rng.uniform(0.3, 0.8)
            pos = np.array([x, y, sz / 2])
            geom = "box"
            size = np.array([sx, sy, sz / 2])
            rgba = np.array([0.6, 0.4, 0.2, 1.0])
        elif kind == "column":
            r = rng.uniform(0.05, 0.15)
            pos = np.array([x, y, zh / 2])
            geom = "cylinder"
            size = np.array([r, zh / 2])
            rgba = np.array([0.7, 0.7, 0.7, 1.0])
        else:  # shelf
            sx = rng.uniform(0.1, 0.3)
            pos = np.array([x, y, rng.uniform(0.5, zh * 0.8)])
            geom = "box"
            size = np.array([sx, 0.05, 0.15])
            rgba = np.array([0.5, 0.3, 0.1, 1.0])

        if _try_place(pos, placed, config):
            obstacles.append(Obstacle(geom, pos, size, rgba))
            placed.append(pos)

    return obstacles


def _generate_random(config: ObstacleConfig, rng: np.random.Generator) -> List[Obstacle]:
    """Generate random mix of obstacles."""
    obstacles = []
    placed = []
    xh, yh, zh = config.arena_size
    attempts = 0

    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        z = rng.uniform(0.1, zh)
        pos = np.array([x, y, z])

        if not _try_place(pos, placed, config):
            continue

        geom = rng.choice(["box", "cylinder", "sphere"])
        if geom == "box":
            size = rng.uniform(0.05, 0.25, size=3)
        elif geom == "cylinder":
            size = np.array([rng.uniform(0.03, 0.15), rng.uniform(0.1, 0.4)])
        else:
            size = np.array([rng.uniform(0.05, 0.2)])

        rgba = np.concatenate([rng.uniform(0.2, 0.9, size=3), [1.0]])
        euler = rng.uniform(-0.5, 0.5, size=3)
        obstacles.append(Obstacle(geom, pos, size, rgba, euler))
        placed.append(pos)

    return obstacles


def _generate_gates(config: ObstacleConfig, rng: np.random.Generator) -> List[Obstacle]:
    """Generate racing gates (rectangular openings)."""
    obstacles = []
    n_gates = min(config.num_obstacles, 10)
    xh, yh, zh = config.arena_size

    for i in range(n_gates):
        # Place gates along a path
        angle = 2 * np.pi * i / n_gates
        radius = min(xh, yh) * 0.6
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)
        gate_h = rng.uniform(0.8, 1.5)
        gate_w = rng.uniform(0.4, 0.8)
        thickness = 0.03

        # Gate is made of 3 boxes: left pillar, right pillar, top bar
        yaw = angle + np.pi / 2  # gate faces inward

        # Simplified: place gate as 3 obstacles (frame)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        # Left pillar
        lx = cx + gate_w / 2 * cos_y
        ly = cy + gate_w / 2 * sin_y
        obstacles.append(Obstacle(
            "box", np.array([lx, ly, gate_h / 2]),
            np.array([thickness, thickness, gate_h / 2]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))
        # Right pillar
        rx = cx - gate_w / 2 * cos_y
        ry = cy - gate_w / 2 * sin_y
        obstacles.append(Obstacle(
            "box", np.array([rx, ry, gate_h / 2]),
            np.array([thickness, thickness, gate_h / 2]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))
        # Top bar
        obstacles.append(Obstacle(
            "box", np.array([cx, cy, gate_h]),
            np.array([thickness, gate_w / 2, thickness]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))

    return obstacles
