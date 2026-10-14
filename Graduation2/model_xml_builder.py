import os
import math
import numpy as np
import config_mjx as cfg

def generate_static_mjx_xml():
    xml = f"""<mujoco model="drone_mjx_curriculum">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="{1.0 / cfg.SIM_FREQ}" gravity="0 0 -{cfg.GRAVITY}" iterations="10" solver="CG"/>

  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="1.0 0.1 0.1"/>
  </default>

  <visual>
    <global offwidth="64" offheight="64"/>
  </visual>

  <worldbody>
    <light directional="true" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 25 0.1" rgba="0.85 0.85 0.85 1" pos="0 0 0"/>

    <body name="cf2" pos="0 11 1">
      <freejoint name="root"/>
      <geom name="cf2_body" type="sphere" size="0.06" rgba="0.1 0.3 0.9 1" mass="{cfg.CF2_MASS}"/>
      <site name="drone_center" pos="0 0 0"/>
      <camera name="fpv_cam" pos="0.04 0 0.02" euler="1.57 0 -1.57" fovy="75"/>
    </body>

    <body name="goal_marker" pos="0 7.5 1.5" mocap="true">
      <geom name="goal_geom" type="sphere" size="{cfg.GOAL_THRESHOLD}" rgba="1 0 0 0.4" contype="0" conaffinity="0"/>
    </body>
"""

    np.random.seed(42)
    for i, (ox, oy) in enumerate(cfg.FIXED_OBSTACLE_POSITIONS):
        xml += f"""
    <body name="obs_{i}" pos="{ox} {oy} 0">
      <geom name="obs_col_{i}" type="cylinder" size="{cfg.CYLINDER_RADIUS} {cfg.CYLINDER_HEIGHT / 2}" pos="0 0 {cfg.CYLINDER_HEIGHT / 2}" rgba="0.6 0.35 0.15 1"/>
"""
        num_spikes = 8
        for s in range(num_spikes):
            z_s = 0.5 + s * (1.8 / num_spikes)
            angle = (s * 2.39996) % (2 * math.pi)
            sp_len = 0.5
            dx = (cfg.CYLINDER_RADIUS + sp_len / 2) * math.cos(angle)
            dy = (cfg.CYLINDER_RADIUS + sp_len / 2) * math.sin(angle)
            xml += f"""      <geom name="spike_{i}_{s}" type="capsule" size="0.02 {sp_len/2}" pos="{dx} {dy} {z_s}" euler="1.57 {angle} 0" rgba="0.8 0.2 0.1 1"/>\n"""
        xml += "    </body>\n"

    xml += """  </worldbody>
  <actuator>
    <motor name="thrust_x" gear="1 0 0 0 0 0" site="drone_center" ctrllimited="true" ctrlrange="-2 2"/>
    <motor name="thrust_y" gear="0 1 0 0 0 0" site="drone_center" ctrllimited="true" ctrlrange="-2 2"/>
    <motor name="thrust_z" gear="0 0 1 0 0 0" site="drone_center" ctrllimited="true" ctrlrange="-2 2"/>
  </actuator>
</mujoco>"""
    return xml
