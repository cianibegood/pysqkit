from typing import Optional

import numpy as np
from .basis import OperatorBasis
from ..operators import low_op, raise_op, num_op, id_op


class FockBasis(OperatorBasis):
    """
    A general Fock basis object, which encodes the Hamiltonian dimensionality (cutoff) as well as the operators necessary (creation operators) and charge/flux operators expressed in this basis.

    TODO: Perhaps one can include commonly used operators, like cos(flux_op) in here.
    TODO: __eq__ and __str__ method to be implemented as well.
    """

    def __init__(self, dim_hilbert):
        super().__init__(dim_hilbert)

    @property
    def low_op(self) -> np.ndarray:
        return low_op(self.dim_hilbert)

    @property
    def raise_op(self) -> np.ndarray:
        return raise_op(self.dim_hilbert)

    @property
    def num_op(self) -> np.ndarray:
        return num_op(self.dim_hilbert)

    @property
    def id_op(self) -> np.ndarray:
        return id_op(self.dim_hilbert)

    def charge_op(self, charge_zpf: Optional[float] = 1.0) -> np.ndarray:
        charge_op = 1j * charge_zpf * (self.raise_op - self.low_op)
        return charge_op

    def flux_op(self, flux_zpf: Optional[float] = 1.0) -> np.ndarray:
        flux_op = flux_zpf * (self.raise_op + self.low_op)
        return flux_op

    def __repr__(self):
        label_str = "Fock state basis, hilbert dim={}".format(self.dim_hilbert)
        return label_str


def fock_basis(dim_hilbert: int):
    if not isinstance(dim_hilbert, int) or dim_hilbert <= 0:
        raise ValueError("Hilbert dimensionality must be a positive integer")
    return FockBasis(dim_hilbert)
