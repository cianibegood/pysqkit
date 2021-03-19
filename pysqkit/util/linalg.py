
from typing import Union, Tuple
import numpy as np


def order_vecs(
    val_vector: np.ndarray,
    pair_vector: np.ndarray = None,
    invert: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(val_vector, np.ndarray):
        raise ValueError("Input vector must be a np.ndarray object")
    if len(val_vector.shape) > 1:
        raise ValueError("Input must be a 1-dimensional vector")

    if pair_vector is not None:
        if not isinstance(pair_vector, np.ndarray):
            raise ValueError("The pair vector must be a np.ndarray object")
        if pair_vector.shape[0] != val_vector.size:
            raise ValueError(
                "The pair vector must have the same dimensionality along the first dimension as the input vector")

    sorted_inds = np.argsort(val_vector)
    if invert:
        sorted_inds = sorted_inds[::-1]

    if pair_vector is not None:
        return val_vector[sorted_inds], pair_vector[sorted_inds]
    else:
        return val_vector[sorted_inds]


def get_mat_elem(
    operator: np.ndarray,
    in_states: np.ndarray,
    out_states: np.ndarray,
):
    in_dim = len(in_states.shape)
    out_dim = len(out_states.shape)

    if len(operator.shape) != 2:
        raise ValueError("The operators must a be a 2D array")

    out_inds = list(range(out_dim))
    in_inds = list(range(out_dim, in_dim + out_dim))
    op_inds = [out_dim - 1, in_dim + out_dim - 1]

    mat_elem = np.einsum(
        out_states, out_inds,
        operator, op_inds,
        in_states, in_inds)
    return mat_elem
