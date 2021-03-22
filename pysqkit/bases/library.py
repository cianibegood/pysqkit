import numpy as np
from .basis import OperatorBasis
from ..operators import low_op, raise_op, num_op


class FockBasis(OperatorBasis):
    """
    A general Fock basis object, which encodes the Hamiltonian dimensionality (cutoff) as well as the operators necessary (creation operators) and charge/flux operators expressed in this basis.

    TODO: Perhaps one can include commonly used operators, like cos(flux_op) in here.
    TODO: __eq__ and __str__ method to be implemented as well.
    """

    def __init__(self, dim_hilbert: int, osc_len: float):
        super().__init__(dim_hilbert)
        self._len = osc_len

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
    def charge_op(self) -> np.ndarray:
        charge_op = 1j * (self.raise_op - self.low_op) / \
            (self._len * np.sqrt(2))
        return charge_op

    @property
    def flux_op(self) -> np.ndarray:
        flux_op = self._len * (self.raise_op + self.low_op) / np.sqrt(2)
        return flux_op

    def __repr__(self):
        label_str = "Fock state basis, hilbert dim={}".format(self.dim_hilbert)
        return label_str


def fock_basis(dim_hilbert: int, osc_len: float):
    if not isinstance(dim_hilbert, int) or dim_hilbert <= 0:
        raise ValueError("Hilbert dimensionality must be a positive integer")

    if not isinstance(dim_hilbert, int):
        raise ValueError("The effect oscillator length must be a float value")

    return FockBasis(dim_hilbert, osc_len)
