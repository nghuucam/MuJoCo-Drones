# Wind Disturbance Wrapper (WindWrapper) Guide in MuJoCo-Drones

This folder contains the complete testing suite, physical benchmarks, and 3D visualization utilities for **[`WindWrapper`](file:///c:/KLTN/MuJoCo-Drones/multi_drone_mujoco/wrappers/wind_wrapper.py)** – a dedicated Gymnasium wrapper that injects aerodynamic wind disturbances and turbulence directly into the MuJoCo multi-substep physics engine.

---

## 1. Directory Structure

```
c:\KLTN\MuJoCo-Drones\Test\aviary_demos3\
├── controller_utils3.py
├── 01_test_wind_constant.py
├── 02_test_wind_gust.py
├── 03_test_wind_dryden.py
├── 04_test_wind_sinusoidal.py
├── 05_test_wind_combined.py
├── 06_compare_wind_vs_nowind.py
├── 07_test_wind_drift_visible.py
├── 08_test_wind_swirls_icon.py
├── run_all_wind.py
├── README.md
└── gifs/
    ├── wind_constant.gif
    ├── wind_gust.gif
    ├── wind_dryden.gif
    ├── wind_sinusoidal.gif
    ├── wind_combined.gif
    ├── compare_benchmark.gif
    ├── wind_resist_visible.gif
    ├── wind_drift_visible.gif
    └── wind_swirls_icon.gif
```

---

## 2. Physics Formulation of `WindWrapper`

`WindWrapper` computes aerodynamic drag and relative air velocity forces applied to each drone body at every physics sub-step:

$$\mathbf{F}_{\text{wind}} = \frac{1}{2} \rho \, (C_d \cdot A) \, \|\mathbf{v}_{\text{rel}}\| \, \mathbf{v}_{\text{rel}}$$

Where:
- $\rho = 1.225 \text{ kg/m}^3$: Air density.
- $\mathbf{v}_{\text{rel}} = \mathbf{w}_{\text{wind}} - \mathbf{v}_{\text{drone}}$: Relative velocity between ambient wind field and drone.
- $C_d \cdot A$: Effective cross-sectional aerodynamic drag area.
- $\mathbf{F}_{\text{wind}}$ is integrated at 240Hz sub-steps inside `BaseAviary._physics` via MuJoCo external force buffer `data.xfrc_applied`.

---

## 3. Supported Wind Models

| Model | Primary Parameters | Physical Characteristics & Drone Response |
| :--- | :--- | :--- |
| **`CONSTANT`** | `constant_wind=[wx, wy, wz]` | Continuous steady wind. Drone maintains constant tilt angle (Pitch/Roll) to counter drift. |
| **`GUST`** | `gust_intensity`, `gust_probability`, `gust_duration_steps` | Stochastic discrete wind impulses. Tests controller recovery and disturbance rejection. |
| **`DRYDEN`** | `turbulence_intensity`, `altitude`, `airspeed` | Military aviation standard (MIL-F-8785C). Continuous multi-axis frequency-shaped turbulence. |
| **`SINUSOIDAL`** | `sinusoidal_amplitude`, `sinusoidal_period` | Harmonic oscillatory wind $F(t) = A \sin(2\pi t / T)$. Drone sways periodically. |
| **`COMBINED`** | Superposition of all components | Extreme weather benchmark: Steady wind + Dryden turbulence + Stochastic gusts. |

---

## 4. Running Demonstrations

Navigate to the test directory:
```powershell
cd C:\KLTN\MuJoCo-Drones\Test\aviary_demos3
```

### Interactive 3D Viewer (`--gui`)

```powershell
python 01_test_wind_constant.py --gui
python 02_test_wind_gust.py --gui
python 03_test_wind_dryden.py --gui
python 04_test_wind_sinusoidal.py --gui
python 05_test_wind_combined.py --gui
python 06_compare_wind_vs_nowind.py
```

### Parameter Tuning

```powershell
python 01_test_wind_constant.py --gui --wind-speed 3.0
python 03_test_wind_dryden.py --gui --turbulence 2.0
python 02_test_wind_gust.py --gui --gust-intensity 0.02
```

### Run All Wind Models via `run_all_wind.py`

```powershell
python run_all_wind.py
python run_all_wind.py --record
python run_all_wind.py --model constant --gui
```

---

## 5. Visible Wind Experiment

Demonstrates two distinct stages: calm wind (steps 0-70) followed by a 6 m/s wind force:

```powershell
python 07_test_wind_drift_visible.py --gui --mode drift
python 07_test_wind_drift_visible.py --gui --mode resist
```

---

## 6. 3D Aerodynamic Wind Gust Swirls

Visualizes animated 3D aerodynamic wind streamlines with looped curls curling along the active wind vector:

```powershell
python 08_test_wind_swirls_icon.py --gui
python run_all_wind.py --model swirls --gui
```
