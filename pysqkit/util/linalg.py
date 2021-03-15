
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
    if np.all(sorted_inds == np.arange(val_vector.size)):
        if pair_vector is not None:
            return val_vector, pair_vector
        else:
            return val_vector
    else:
        if pair_vector is not None:
            return val_vector[sorted_inds], pair_vector[sorted_inds]
        else:
            return val_vector[sorted_inds]
