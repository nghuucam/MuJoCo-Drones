import os
import sys
import numpy as np

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

try:
    import jax
    import jax.numpy as jnp
    from jax.dlpack import from_dlpack, to_dlpack
except ImportError:
    pass

import config_mjx as cfg


class WarpBatchRenderer:
    def __init__(self, num_envs: int = cfg.NUM_ENVS, width: int = 64, height: int = 64):
        self.num_envs = num_envs
        self.width = width
        self.height = height

        if HAS_WARP:
            wp.init()
            self.device = "cuda" if wp.is_cuda_available() else "cpu"
            self.pixels = wp.zeros((num_envs, height, width, 3), dtype=wp.uint8, device=self.device)
        else:
            self.pixels = None

    def render_batch(self, drone_positions: jnp.ndarray, drone_orientations: jnp.ndarray) -> jnp.ndarray:
        if not HAS_WARP:
            return jnp.zeros((self.num_envs, self.height, self.width, 3), dtype=jnp.uint8)

        dl_tensor = wp.to_dlpack(self.pixels)
        jax_images = from_dlpack(dl_tensor)
        return jax_images
