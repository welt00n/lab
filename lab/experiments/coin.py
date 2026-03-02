"""Coin drop experiment — thin declarative subclass."""

import math

from lab.core.rigid_body_jit import COIN, COIN_RADIUS
from lab.experiments.base import DropExperiment


class CoinDrop(DropExperiment):
    shape = "coin"
    shape_id = COIN
    angle_range = (0, 2 * math.pi)
    colors = {1: "#1f77b4", -1: "#d62728", 0: "#cccccc"}
    labels = {1: "Heads", -1: "Tails", 0: "Edge"}
    settle_height = 5.0 * COIN_RADIUS
    body_color = "#7799bb"
    mesh = "coin"
