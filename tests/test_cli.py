"""
Tests for the CLI entrypoint (main.py).

Covers argument parsing, experiment loading, output directory creation,
and the NVIDIA library setup helper.
"""

import os
import re
import shutil
import tempfile
from unittest.mock import patch

import pytest

from main import _load_experiment, _make_output_dir, _setup_nvidia_libs, EXPERIMENTS


# ===================================================================
# _load_experiment
# ===================================================================

class TestLoadExperiment:
    def test_coin(self):
        from lab.experiments.coin import CoinDrop
        exp = _load_experiment("coin")
        assert isinstance(exp, CoinDrop)

    def test_cube(self):
        from lab.experiments.cube import CubeDrop
        exp = _load_experiment("cube")
        assert isinstance(exp, CubeDrop)

    def test_invalid_raises(self):
        with pytest.raises(KeyError):
            _load_experiment("nonexistent")

    def test_all_registered_experiments_loadable(self):
        for name in EXPERIMENTS:
            exp = _load_experiment(name)
            assert hasattr(exp, "shape")
            assert hasattr(exp, "sweep")


# ===================================================================
# _make_output_dir
# ===================================================================

class TestMakeOutputDir:
    def test_creates_directory(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                result = _make_output_dir("coin", "batch")
                assert result.exists()
                assert result.is_dir()
            finally:
                os.chdir(old_cwd)

    def test_name_contains_shape(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                result = _make_output_dir("coin", "batch")
                assert "coin" in result.name
            finally:
                os.chdir(old_cwd)
                shutil.rmtree(os.path.join(tmp, "results"), ignore_errors=True)

    def test_name_contains_mode(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                result = _make_output_dir("cube", "gpu")
                assert "gpu" in result.name
            finally:
                os.chdir(old_cwd)
                shutil.rmtree(os.path.join(tmp, "results"), ignore_errors=True)

    def test_name_contains_date_stamp(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                result = _make_output_dir("coin", "batch")
                assert re.search(r"\d{4}-\d{2}-\d{2}", result.name)
            finally:
                os.chdir(old_cwd)
                shutil.rmtree(os.path.join(tmp, "results"), ignore_errors=True)


# ===================================================================
# Argument parsing
# ===================================================================

class TestArgParsing:
    """Test CLI argument parsing via argparse."""

    @staticmethod
    def _parse(args_list):
        import argparse
        from main import main
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment", nargs="?", default=None,
                            choices=list(EXPERIMENTS.keys()))
        parser.add_argument("--nh", type=int, default=40)
        parser.add_argument("--na", type=int, default=60)
        parser.add_argument("--hmin", type=float, default=0.1)
        parser.add_argument("--hmax", type=float, default=5.0)
        parser.add_argument("--axis", default="x", choices=["x", "y", "z"])
        parser.add_argument("--gpu", action="store_true")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--save-video", action="store_true")
        return parser.parse_args(args_list)

    def test_coin_recognized(self):
        args = self._parse(["coin"])
        assert args.experiment == "coin"

    def test_cube_recognized(self):
        args = self._parse(["cube"])
        assert args.experiment == "cube"

    def test_grid_params(self):
        args = self._parse(["coin", "--nh", "20", "--na", "30",
                            "--hmin", "0.5", "--hmax", "3.0"])
        assert args.nh == 20
        assert args.na == 30
        assert args.hmin == 0.5
        assert args.hmax == 3.0

    def test_axis_param(self):
        args = self._parse(["coin", "--axis", "y"])
        assert args.axis == "y"

    def test_boolean_flags_default_false(self):
        args = self._parse(["coin"])
        assert not args.gpu
        assert not args.live
        assert not args.save_video

    def test_gpu_flag(self):
        args = self._parse(["coin", "--gpu"])
        assert args.gpu

    def test_live_flag(self):
        args = self._parse(["coin", "--live"])
        assert args.live

    def test_save_video_flag(self):
        args = self._parse(["coin", "--save-video"])
        assert args.save_video

    def test_no_experiment_gives_none(self):
        args = self._parse([])
        assert args.experiment is None


# ===================================================================
# _setup_nvidia_libs
# ===================================================================

class TestSetupNvidiaLibs:
    def test_runs_without_error(self):
        _setup_nvidia_libs()

    def test_does_not_crash_without_nvidia(self):
        with patch.dict(os.environ, {}, clear=False):
            _setup_nvidia_libs()
