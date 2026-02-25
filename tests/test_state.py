"""Unit tests for State."""

import numpy as np
from lab.core.state import State


class TestState:
    def test_creation(self):
        s = State([1, 2], [3, 4])
        assert s.ndof == 2
        np.testing.assert_array_equal(s.q, [1.0, 2.0])
        np.testing.assert_array_equal(s.p, [3.0, 4.0])

    def test_copy_is_independent(self):
        s = State([1.0], [2.0])
        c = s.copy()
        c.q[0] = 99.0
        assert s.q[0] == 1.0

    def test_stores_as_float(self):
        s = State([1, 2], [3, 4])
        assert s.q.dtype == np.float64
        assert s.p.dtype == np.float64

    def test_input_not_mutated(self):
        q = np.array([1.0, 2.0])
        p = np.array([3.0, 4.0])
        s = State(q, p)
        s.q[0] = 99.0
        assert q[0] == 1.0
