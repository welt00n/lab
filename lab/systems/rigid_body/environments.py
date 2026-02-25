"""
Environment presets — named bundles of fields and constraints.
"""

import numpy as np
from lab.systems.rigid_body.fields import (
    GravityField, UniformElectricField, DragField,
)
from lab.systems.rigid_body.constraints import FloorConstraint


class Environment:
    """A named collection of fields and constraints."""

    def __init__(self, name, fields=None, constraints=None):
        self.name = name
        self.fields = fields or []
        self.constraints = constraints or []

    def apply(self, world):
        for f in self.fields:
            world.add_field(f)
        for c in self.constraints:
            world.add_constraint(c)

    def __repr__(self):
        return (f"Environment({self.name!r}, "
                f"fields={len(self.fields)}, constraints={len(self.constraints)})")


def earth_surface(restitution=0.8):
    return Environment("earth_surface",
                       fields=[GravityField(9.81)],
                       constraints=[FloorConstraint(restitution=restitution)])

def vacuum(g=9.81):
    return Environment("vacuum", fields=[GravityField(g)])

def capacitor(E_strength=1000.0, restitution=0.8):
    return Environment("capacitor",
                       fields=[GravityField(9.81),
                               UniformElectricField(np.array([0.0, E_strength, 0.0]))],
                       constraints=[FloorConstraint(restitution=restitution)])
