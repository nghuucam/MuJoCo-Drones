# Procedural Obstacle Environments Guide in MuJoCo-Drones

This folder contains the complete test scripts, 3D visualizations, and GIF export utilities for the **Procedural Obstacle Generation** system with 5 distinct environment themes:

1. **`FOREST`** (Woodland: Cylindrical tree trunks + Spherical foliage crowns)
2. **`URBAN`** (Metropolitan Canyon: Concrete buildings and red warning beacons)
3. **`INDOOR`** (Enclosed Room: 4 boundary walls, ceiling, tables, and pillars)
4. **`RANDOM`** (Cluttered Benchmark: Randomly oriented spheres, cylinders, and boxes)
5. **`GATES`** (FPV Slalom Course: Sequential racing gates with portal indicators)

---

## 1. Directory Structure

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos2\
├── controller_utils2.py
├── 01_test_obstacle_forest.py
├── 02_test_obstacle_urban.py
├── 03_test_obstacle_indoor.py
├── 04_test_obstacle_random.py
├── 05_test_obstacle_gates.py
├── run_all_obstacles.py
├── README.md
└── gifs/
    ├── obstacle_forest.gif
    ├── obstacle_urban.gif
    ├── obstacle_indoor.gif
    ├── obstacle_random.gif
    └── obstacle_gates.gif
```

---

## 2. Technical Specifications of Obstacle Themes

| Theme | Core Geometries | Materials & Colors | Spatial Characteristics & Challenge |
| :--- | :--- | :--- | :--- |
| **`FOREST`** | Trunk `cylinder`, Crown `sphere` | Bark brown, natural green canopy | Dense spacing, canopy downwash, agility through tree trunks. |
| **`URBAN`** | Building `box`, Beacon `sphere` | Concrete gray, stone, glass, red beacon | Urban canyons, sheer vertical surfaces, street intersections. |
| **`INDOOR`** | 4 Walls `box`, Ceiling `box`, Pillars | Ivory walls, brown wood, gray pillars | Enclosed volume with ceiling altitude constraint and ground effect. |
| **`RANDOM`** | Mixed `box`, `cylinder`, `sphere` | Multi-color palette, random Euler angles | Unstructured clutter evaluating RL policy generalization. |
| **`GATES`** | 2 Posts `box/cylinder`, Crossbar, Portal | Neon orange, reflective yellow, cyan center | Sequential narrow apertures testing precision flight. |

---

## 3. Running Demonstrations

Navigate to the test directory:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos2
```

### Interactive 3D Viewer (`--gui`)

```powershell
python 01_test_obstacle_forest.py --gui
python 02_test_obstacle_urban.py --gui
python 03_test_obstacle_indoor.py --gui
python 04_test_obstacle_random.py --gui
python 05_test_obstacle_gates.py --gui
```

### Record Animated GIFs (`--record`)

```powershell
python 01_test_obstacle_forest.py --record
python 02_test_obstacle_urban.py --record
python 03_test_obstacle_indoor.py --record
python 04_test_obstacle_random.py --record
python 05_test_obstacle_gates.py --record
```

### Run All Themes via `run_all_obstacles.py`

```powershell
python run_all_obstacles.py
python run_all_obstacles.py --record
python run_all_obstacles.py --theme forest --gui
```

### Configuration Options
- `--num-obstacles <N>`: Adjust obstacle count (e.g. `--num-obstacles 40`).
- `--seed <S>`: Change random seed for procedural map generation.
- `--steps <K>`: Total simulation steps.
