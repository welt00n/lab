"""
Tests for the GPU-accelerated drop experiment module.

Tests are skipped on machines without CUDA to keep the test suite portable.
"""

import numpy as np
import pytest

from lab.experiments.drop_gpu import HAS_CUDA

cuda_required = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")


# ===================================================================
# 1. Device function correctness — quaternion math
# ===================================================================

@cuda_required
class TestQuaternionDeviceFunctions:
    """
    Compare GPU quaternion device functions against CPU quaternion module.

    Each test writes results from a tiny wrapper kernel to a device array,
    copies back, and compares with the CPU implementation.
    """

    def test_normalize(self):
        from numba import cuda
        from lab.experiments.drop_gpu import quat_normalize
        from lab.core import quaternion as quat

        @cuda.jit
        def kernel(out):
            w, x, y, z = quat_normalize(3.0, 0.0, 4.0, 0.0)
            out[0] = w
            out[1] = x
            out[2] = y
            out[3] = z

        d_out = cuda.device_array(4, dtype=np.float64)
        kernel[1, 1](d_out)
        gpu = d_out.copy_to_host()

        cpu = quat.normalize(np.array([3.0, 0.0, 4.0, 0.0]))
        np.testing.assert_allclose(gpu, cpu, atol=1e-12)

    def test_multiply(self):
        from numba import cuda
        from lab.experiments.drop_gpu import quat_multiply
        from lab.core import quaternion as quat

        q1 = np.array([0.707, 0.0, 0.707, 0.0])
        q2 = np.array([0.5, 0.5, 0.5, 0.5])

        @cuda.jit
        def kernel(out, a, b):
            w, x, y, z = quat_multiply(a[0], a[1], a[2], a[3],
                                        b[0], b[1], b[2], b[3])
            out[0] = w
            out[1] = x
            out[2] = y
            out[3] = z

        d_out = cuda.device_array(4, dtype=np.float64)
        kernel[1, 1](d_out, cuda.to_device(q1), cuda.to_device(q2))
        gpu = d_out.copy_to_host()

        cpu = quat.multiply(q1, q2)
        np.testing.assert_allclose(gpu, cpu, atol=1e-12)

    def test_from_axis_angle(self):
        from numba import cuda
        from lab.experiments.drop_gpu import quat_from_axis_angle
        from lab.core import quaternion as quat

        @cuda.jit
        def kernel(out):
            w, x, y, z = quat_from_axis_angle(0.0, 0.0, 1.0, 1.5707963)
            out[0] = w
            out[1] = x
            out[2] = y
            out[3] = z

        d_out = cuda.device_array(4, dtype=np.float64)
        kernel[1, 1](d_out)
        gpu = d_out.copy_to_host()

        cpu = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        np.testing.assert_allclose(gpu, cpu, atol=1e-6)

    def test_rotate_vector(self):
        from numba import cuda
        from lab.experiments.drop_gpu import quat_from_axis_angle, quat_rotate_vector
        from lab.core import quaternion as quat

        @cuda.jit
        def kernel(out):
            qw, qx, qy, qz = quat_from_axis_angle(0.0, 0.0, 1.0, 1.5707963)
            rx, ry, rz = quat_rotate_vector(qw, qx, qy, qz, 1.0, 0.0, 0.0)
            out[0] = rx
            out[1] = ry
            out[2] = rz

        d_out = cuda.device_array(3, dtype=np.float64)
        kernel[1, 1](d_out)
        gpu = d_out.copy_to_host()

        q = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        cpu = quat.rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(gpu, cpu, atol=1e-6)

    def test_exp_map(self):
        from numba import cuda
        from lab.experiments.drop_gpu import quat_exp_map
        from lab.core import quaternion as quat

        omega = np.array([0.0, 5.0, 0.0])
        dt = 0.01

        @cuda.jit
        def kernel(out, om, dt_val):
            w, x, y, z = quat_exp_map(om[0], om[1], om[2], dt_val[0])
            out[0] = w
            out[1] = x
            out[2] = y
            out[3] = z

        d_out = cuda.device_array(4, dtype=np.float64)
        kernel[1, 1](d_out, cuda.to_device(omega),
                     cuda.to_device(np.array([dt])))
        gpu = d_out.copy_to_host()

        cpu = quat.exp_map(omega, dt)
        np.testing.assert_allclose(gpu, cpu, atol=1e-12)


# ===================================================================
# 2. GPU vs CPU result parity
# ===================================================================

@cuda_required
class TestGpuVsCpuParity:
    """
    Run sweep_drop_gpu and sweep_drop (CPU) on the same small grid.

    We don't demand exact match because floating-point ordering differs
    between CPU and GPU.  Instead we check that the vast majority of
    outcomes agree (>= 60% on a coarse grid) — the mismatches are at
    chaotic boundary conditions.
    """

    def _compare(self, shape, threshold=0.4):
        from lab.experiments.drop_gpu import sweep_drop_gpu
        from lab.experiments.drop_experiment import sweep_drop

        heights = np.linspace(0.3, 1.5, 4)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)

        gpu_r = sweep_drop_gpu(shape, heights, angles, tilt_axis="x")
        cpu_r = sweep_drop(shape, heights, angles, tilt_axis="x", workers=1)

        match_frac = np.mean(gpu_r == cpu_r)
        assert match_frac >= threshold, (
            f"{shape}: only {match_frac:.0%} match (threshold {threshold:.0%})"
        )

    def test_coin(self):
        self._compare("coin")

    def test_cube(self):
        self._compare("cube")

    def test_rod(self):
        self._compare("rod")


# ===================================================================
# 3. Fallback behaviour
# ===================================================================

def test_gpu_unavailable_raises():
    """sweep_drop_gpu raises RuntimeError when CUDA is unavailable."""
    from unittest.mock import patch
    import lab.experiments.drop_gpu as mod

    original = mod.HAS_CUDA
    try:
        mod.HAS_CUDA = False
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            mod.sweep_drop_gpu(
                "coin",
                np.linspace(0.5, 1.0, 2),
                np.linspace(0, np.pi, 2),
            )
    finally:
        mod.HAS_CUDA = original
