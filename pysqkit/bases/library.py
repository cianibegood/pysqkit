import numpy as np
from .basis import OperatorBasis
from ..operators import low_op, raise_op, num_op


class FockBasis(OperatorBasis):
    """
    A general Fock basis object, which encodes the Hamiltonian dimensionality 
    (cutoff) as well as the operators necessary (creation operators) and charge/flux 
    operators expressed in this basis.

    TODO: Perhaps one can include commonly used operators, like cos(flux_op) in here.
    TODO: __eq__ and __str__ method to be implemented as well.
    """

    @property
    def low_op(self) -> np.ndarray:
        return low_op(self.dim_hilbert)

    @property
    def raise_op(self) -> np.ndarray:
        return raise_op(self.dim_hilbert)

    @property
    def num_op(self) -> np.ndarray:
        return num_op(self.dim_hilbert)

    def __repr__(self) -> str:
        label_str = "Fock state basis, hilbert dim={}".format(self.dim_hilbert)
        return label_str


def fock_basis(dim_hilbert: int):
    if not isinstance(dim_hilbert, int) or dim_hilbert <= 0:
        raise ValueError("Hilbert dimensionality must be a positive integer")

    return FockBasis(dim_hilbert)
