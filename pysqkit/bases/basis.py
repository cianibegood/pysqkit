from abc import ABC

import numpy as np

from ..operators import id_op


class OperatorBasis(ABC):
    """
    NOTE: This class is a bit empty currently. We can remove it - I added it as we might in theory look at sparse operator, in which case that would be integrated here I think.
    """

    def __init__(self, dim_hilbert: int):
        self._dim_hilbert = dim_hilbert
        self._transformation = None

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
        self._dim_hilbert = new_dim

    @property
    def id_op(self) -> np.ndarray:
        return id_op(self.dim_hilbert)

    @property
    def transformation(self) -> np.ndarray:
        return self._transformation

    def transform(self, transform_mat: np.ndarray) -> None:
        if not isinstance(transform_mat, np.ndarray):
            raise ValueError(
                "The transformation matrix should be provided as"
                "a np.ndarry object, got {} instead".format(
                    type(transform_mat)
                )
            )

        if len(transform_mat.shape) != 2:
            raise ValueError(
                "The transformaton matrix should be a 2-dimensional matrix, "
                "instead got a {} one".format(
                    len(transform_mat.shape)
                )
            )

            self._transformation = transform_mat
