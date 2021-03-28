from abc import ABC

import numpy as np

from ..operators import id_op


class OperatorBasis(ABC):
    """
    NOTE: This class is a bit empty currently. We can remove it - I added it as we might in theory look at sparse operator, in which case that would be integrated here I think.
    """

    def __init__(self, dim_hilbert):
        self._dim_hilbert = dim_hilbert

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
        self._dim_hilbert = new_dim

    @property
    def id_op(self) -> np.ndarray:
        return id_op(self.dim_hilbert)
