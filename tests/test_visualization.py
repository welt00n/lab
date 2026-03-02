"""
Tests for the generic visualization primitives: sweep_grid, category_histogram.

Uses matplotlib Agg backend for headless rendering.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.container import BarContainer

from lab.visualization import sweep_grid, category_histogram


# ===================================================================
# sweep_grid._to_rgb
# ===================================================================

class TestToRgb:
    def test_uniform_grid_single_colour(self):
        data = np.ones((3, 4), dtype=int)
        colors = {1: "#ff0000"}
        rgb = sweep_grid._to_rgb(data, colors)
        assert rgb.shape == (3, 4, 3)
        np.testing.assert_allclose(rgb[0, 0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_mixed_grid(self):
        data = np.array([[0, 1], [1, 0]])
        colors = {0: "#ff0000", 1: "#0000ff"}
        rgb = sweep_grid._to_rgb(data, colors)
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(rgb[0, 1], [0, 0, 1], atol=1e-6)

    def test_nan_uses_default_colour(self):
        data = np.full((2, 2), np.nan)
        colors = {1: "#ff0000"}
        nan_c = (0.85, 0.85, 0.85)
        rgb = sweep_grid._to_rgb(data, colors, nan_color=nan_c)
        np.testing.assert_allclose(rgb[0, 0], nan_c)

    def test_empty_colour_map(self):
        data = np.array([[1, 2], [3, 4]])
        rgb = sweep_grid._to_rgb(data, {})
        np.testing.assert_allclose(rgb[0, 0], [0.85, 0.85, 0.85])


# ===================================================================
# sweep_grid.create / update
# ===================================================================

class TestSweepGridCreate:
    def test_returns_axes_image(self):
        fig, ax = plt.subplots()
        data = np.zeros((5, 5), dtype=int)
        img = sweep_grid.create(data, {0: "#aaa"}, [0, 1, 0, 1],
                                "x", "y", "title", ax)
        assert isinstance(img, AxesImage)
        plt.close(fig)

    def test_image_data_shape(self):
        fig, ax = plt.subplots()
        data = np.zeros((3, 7), dtype=int)
        img = sweep_grid.create(data, {0: "#aaa"}, [0, 1, 0, 1],
                                "x", "y", "title", ax)
        h, w, _ = img.get_array().shape
        assert (h, w) == (3, 7)
        plt.close(fig)


class TestSweepGridUpdate:
    def test_update_changes_data(self):
        fig, ax = plt.subplots()
        data = np.zeros((3, 3), dtype=int)
        colors = {0: "#ff0000", 1: "#0000ff"}
        img = sweep_grid.create(data, colors, [0, 1, 0, 1],
                                "x", "y", "t", ax)
        new_data = np.ones((3, 3), dtype=int)
        sweep_grid.update(img, new_data, colors)
        rgb = img.get_array()
        np.testing.assert_allclose(rgb[0, 0], [0, 0, 1], atol=1e-6)
        plt.close(fig)


# ===================================================================
# category_histogram.create / update
# ===================================================================

class TestCategoryHistogramCreate:
    def test_returns_bar_container(self):
        fig, ax = plt.subplots()
        cats = [0, 1, 2]
        colors = {0: "#f00", 1: "#0f0", 2: "#00f"}
        labels = {0: "a", 1: "b", 2: "c"}
        bars = category_histogram.create(cats, colors, labels, ax)
        assert isinstance(bars, BarContainer)
        plt.close(fig)

    def test_correct_number_of_bars(self):
        fig, ax = plt.subplots()
        cats = [1, -1, 0]
        colors = {1: "#f00", -1: "#0f0", 0: "#00f"}
        labels = {1: "H", -1: "T", 0: "E"}
        bars = category_histogram.create(cats, colors, labels, ax)
        assert len(bars) == 3
        plt.close(fig)

    def test_bars_start_at_zero(self):
        fig, ax = plt.subplots()
        cats = [0, 1]
        colors = {0: "#f00", 1: "#0f0"}
        labels = {0: "a", 1: "b"}
        bars = category_histogram.create(cats, colors, labels, ax)
        for bar in bars:
            assert bar.get_height() == 0
        plt.close(fig)


class TestCategoryHistogramUpdate:
    def test_bar_heights_match_counts(self):
        fig, ax = plt.subplots()
        cats = [0, 1]
        colors = {0: "#f00", 1: "#0f0"}
        labels = {0: "a", 1: "b"}
        bars = category_histogram.create(cats, colors, labels, ax)

        data = np.array([[0, 0, 1], [1, 1, 0]])
        category_histogram.update(bars, data, cats)
        heights = [bar.get_height() for bar in bars]
        assert heights == [3, 3]
        plt.close(fig)

    def test_empty_data_all_zero(self):
        fig, ax = plt.subplots()
        cats = [0, 1]
        colors = {0: "#f00", 1: "#0f0"}
        labels = {0: "a", 1: "b"}
        bars = category_histogram.create(cats, colors, labels, ax)

        data = np.full((2, 2), np.nan)
        category_histogram.update(bars, data, cats)
        heights = [bar.get_height() for bar in bars]
        assert heights == [0, 0]
        plt.close(fig)
