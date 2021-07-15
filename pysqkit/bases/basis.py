from abc import ABC
from math import exp
from typing import List

import numpy as np

from ..operators import id_op
from ..util.linalg import tensor_prod, transform_basis


class OperatorBasis(ABC):
    """
    NOTE: This class is a bit empty currently. We can remove it - I added it as we might in theory look at sparse operator, in which case that would be integrated here I think.
    """

    def __init__(self, dim_hilbert: int):
        self._dim_hilbert = dim_hilbert

        self._transformation = None
        self._trunc_dim = None

        self._subsys_ind = None
        self._sys_dims = None

    def __copy__(self):
        basis_copy = self.__class__(self._dim_hilbert)

        basis_copy._transformation = self._transformation
        basis_copy._trunc_dim = self._trunc_dim
        basis_copy._subsys_ind = self._subsys_ind
        basis_copy._sys_dims = self._sys_dims

        return basis_copy

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
            return self._sys_dims
        return [self.truncated_dim]

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
        if self.is_truncated:
            if new_dim <= self._trunc_dim:
                raise ValueError(
                    "The new Hilbert dimensionality must be greater than the"
                    "truncated dimension {}".format(self._trunc_dim)
                )
        self._dim_hilbert = new_dim

    @property
    def id_op(self) -> np.ndarray:
        return id_op(self.dim_hilbert)

    @property
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
                raise ValueError(
                    "The operator must be a square 2D matrix, "
                    "instead got shape {}".format(op.shape)
                )
            if op_dim < self._trunc_dim:
                raise ValueError(
                    "Operator dimensions are smaller than " " the truncated dimension"
                )
            return op[: self._trunc_dim, : self._trunc_dim]
        return op

    def transform_op(self, op: np.ndarray) -> np.ndarray:
        if self.is_transformed:
            op_dim = op.shape[0]
            if op.shape != (op_dim, op_dim) or op_dim != self._dim_hilbert:
                raise ValueError(
                    "Operator expected as a 2D square matrix "
                    "with the same dimensionality as the basis, "
                    "instead got {}".format(op.shape)
                )
            return transform_basis(op, self._transformation)
        return op

    def expand_op(self, op: np.ndarray) -> np.ndarray:
        if self.is_subbasis:
            _ops = [
                op if i == self._subsys_ind else id_op(dim)
                for i, dim in enumerate(self._sys_dims)
            ]
            return tensor_prod(_ops)
        return op

    def finalize_op(
        self,
        op: np.ndarray,
        expand=True,
        truncate=True,
    ) -> np.ndarray:
        final_op = self.transform_op(op)
        if truncate:
            final_op = self.truncate_op(final_op)
        if expand:
            final_op = self.expand_op(final_op)
        return final_op

    def embed(self, subsys_ind: int, sys_dims: List[int]) -> None:
        if not isinstance(subsys_ind, int):
            raise ValueError("The subsystem index provided must be an integer")

        if subsys_ind < 0 or subsys_ind > len(sys_dims):
            raise ValueError(
                "Subsystem index must be an integer between 0 and the"
                "total number of subsystem {}".format(len(sys_dims))
            )

        if sys_dims[subsys_ind] != self.truncated_dim:
            raise ValueError(
                "Mismatch between basis dimension and the "
                "specified system dimensions"
                " (given the provided index)."
            )

        self._subsys_ind = subsys_ind
        self._sys_dims = sys_dims

    def unembed(self) -> None:
        self._subsys_ind = None
        self._sys_dims = None

    def truncate(self, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "The truncated dimension provided must be an integer greater than 0"
            )

        if dim > self._dim_hilbert:
            raise ValueError(
                "The truncated dimension must be smaller than the hilbert dimension of the basis"
            )

        self._trunc_dim = dim

    def transform(self, transform_mat: np.ndarray) -> None:
        if not isinstance(transform_mat, np.ndarray):
            raise ValueError(
                "The transformation matrix should be provided as"
                "a np.ndarry object, got {} instead".format(type(transform_mat))
            )

        dim = transform_mat.shape[0]

        if transform_mat.shape != (dim, dim) or dim != self._dim_hilbert:
            raise ValueError(
                "The transformaton matrix should be a 2D square matrix, "
                "with the same dimensionality as the basis, "
                "instead got a {} dimensional matrix".format(len(transform_mat.shape))
            )

        self._transformation = transform_mat
