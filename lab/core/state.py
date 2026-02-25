"""
State — a point in phase space (q, p).
"""

import numpy as np


class State:
    """
    Holds generalized coordinates and momenta as numpy arrays.

    Parameters
    ----------
    q : array-like
        Generalized coordinates.
    p : array-like
        Conjugate momenta.
    """

    __slots__ = ("q", "p")

    def __init__(self, q, p):
        self.q = np.asarray(q, dtype=float).copy()
        self.p = np.asarray(p, dtype=float).copy()

    @property
    def ndof(self):
        return len(self.q)

    def copy(self):
        return State(self.q.copy(), self.p.copy())

    def __repr__(self):
        return f"State(q={self.q}, p={self.p})"
