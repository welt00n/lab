"""Cube drop experiment — thin declarative subclass."""

import math

from lab.core.rigid_body_jit import CUBE, CUBE_HALF_SIDE
from lab.experiments.base import DropExperiment


class CubeDrop(DropExperiment):
    shape = "cube"
    shape_id = CUBE
    angle_range = (0, 2 * math.pi)
    colors = {
        0: "#1f77b4", 1: "#aec7e8",
        2: "#d62728", 3: "#ff9896",
        4: "#2ca02c", 5: "#98df8a",
    }
    labels = {0: "+x", 1: "-x", 2: "+y", 3: "-y", 4: "+z", 5: "-z"}
    settle_height = 3.0 * CUBE_HALF_SIDE
    body_color = "#bb9977"
    mesh = "cube"
