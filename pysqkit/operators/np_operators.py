import numpy as np


def low_op(dim: int) -> np.ndarray:
    low_op = np.diag(np.sqrt(np.arange(1, dim)), 1)
    return low_op


def raise_op(dim: int) -> np.ndarray:
    return low_op(dim).conj().T


def num_op(dim: int) -> np.ndarray:
    return np.diag(np.arange(dim))


def id_op(dim: int) -> np.ndarray:
    return np.eye(dim)


sigma = {
    'I': np.array([[1., 0.], [0., 1.]], dtype=complex),
    'X': np.array([[0., 1.], [1., 0.]], dtype=complex),
    'Y': np.array([[0., -1j], [1j, 0.]], dtype=complex),
    'Z': np.array([[1., 0.], [0., -1.]], dtype=complex),
}
