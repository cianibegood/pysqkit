from abc import ABC
<<<<<<< HEAD
=======
from typing import List, Union
>>>>>>> development

import numpy as np

from ..operators import id_op
<<<<<<< HEAD
=======
from ..util.linalg import tensor_prod, transform_basis
>>>>>>> development


class OperatorBasis(ABC):
    """
    NOTE: This class is a bit empty currently. We can remove it - I added it as we might in theory look at sparse operator, in which case that would be integrated here I think.
    """

    def __init__(self, dim_hilbert: int):
        self._dim_hilbert = dim_hilbert
<<<<<<< HEAD
        self._transformation = None

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
=======

        self._transformation = None
        self._trunc_dim = None

        self._subsys_ind = None
        self._sys_bases = None

    @property
    def dim_hilbert(self) -> int:
        return self._dim_hilbert

    @property
    def truncated_dim(self) -> int:
        if self.is_truncated:
            return self._trunc_dim
        return self._dim_hilbert

    @property
    def sys_truncated_dims(self) -> List[int]:
        if self.is_subbasis:
            return [basis.truncated_dim for basis in self._sys_bases]
        return [self.truncated_dim]

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
        if self.is_truncated:
            if new_dim <= self._trunc_dim:
                raise ValueError(
                    "The new Hilbert dimensionality must be greater than the"
                    "truncated dimension {}".format(self._trunc_dim))
>>>>>>> development
        self._dim_hilbert = new_dim

    @property
    def id_op(self) -> np.ndarray:
        return id_op(self.dim_hilbert)

    @property
<<<<<<< HEAD
    def transformation(self) -> np.ndarray:
        return self._transformation

=======
    def is_subbasis(self) -> bool:
        return self._subsys_ind is not None

    @property
    def is_transformed(self) -> bool:
        return self._transformation is not None

    @property
    def is_truncated(self) -> bool:
        return self._trunc_dim is not None

    @property
    def transformation(self) -> np.ndarray:
        return self._transformation

    def truncate_op(self, op: np.ndarray) -> np.ndarray:
        if self.is_truncated:
            op_dim = op.shape[0]
            if op.shape != (op_dim, op_dim):
                "The operator must be a square 2D matrix, "
                "instead got shape {}".format(
                    op.shape)
            if op_dim <= self._trunc_dim:
                raise ValueError(
                    "Operator dimensions are smaller than "
                    " the truncated dimension")
            return op[:self._trunc_dim, :self._trunc_dim]
        return op

    def transform_op(self, op: np.ndarray) -> np.ndarray:
        if self.is_transformed:
            op_dim = op.shape[0]
            if op.shape != (op_dim, op_dim) or op_dim != self._dim_hilbert:
                raise ValueError("Operator expected as a 2D square matrix "
                                 "with the same dimensionality as the basis, "
                                 "instead got {}".format(op.shape))
            return transform_basis(op, self._transformation)
        return op

    def expand_op(self, op: np.ndarray) -> np.ndarray:
        if self.is_subbasis:
            _ops = [
                op if b_ind == self._subsys_ind else id_op(basis.truncated_dim)
                for b_ind, basis in enumerate(self._sys_bases)
            ]
            return tensor_prod(_ops)
        return op

    def finalize_op(self, op: np.ndarray) -> np.ndarray:
        return self.expand_op(self.truncate_op(self.transform_op(op)))

    def embed(self, subsys_ind: int, sys_bases: List['OperatorBasis']) -> None:
        if not isinstance(subsys_ind, int):
            raise ValueError("The subsystem index provided must be an integer")
        if subsys_ind < 0 or subsys_ind > len(sys_bases):
            raise ValueError(
                "Subsystem index must be an integer between 0 and the"
                "total number of subsystem {}".format(len(sys_bases)))

        self._subsys_ind = subsys_ind
        self._sys_bases = sys_bases

    def truncate(self, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "The truncated dimension provided must be an integer greater than 0")

        if dim > self._dim_hilbert:
            raise ValueError(
                "The truncated dimension must be smaller than the hilbert dimension of the basis")

        self._trunc_dim = dim

>>>>>>> development
    def transform(self, transform_mat: np.ndarray) -> None:
        if not isinstance(transform_mat, np.ndarray):
            raise ValueError(
                "The transformation matrix should be provided as"
                "a np.ndarry object, got {} instead".format(
                    type(transform_mat)
                )
            )

<<<<<<< HEAD
        if len(transform_mat.shape) != 2:
            raise ValueError(
                "The transformaton matrix should be a 2-dimensional matrix, "
                "instead got a {} one".format(
=======
        basis_dim = transform_mat.shape[0]

        if transform_mat.shape != (basis_dim, basis_dim) or basis_dim != self._dim_hilbert:
            raise ValueError(
                "The transformaton matrix should be a 2D square matrix, "
                "with the same dimensionality as the basis, "
                "instead got a {} dimensional matrix".format(
>>>>>>> development
                    len(transform_mat.shape)
                )
            )

<<<<<<< HEAD
            self._transformation = transform_mat
=======
        self._transformation = transform_mat
>>>>>>> development
