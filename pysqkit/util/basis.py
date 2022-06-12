from itertools import product
from pysqkit.util.linalg import hilbert_schmidt_prod
from typing import List

import numpy as np

pauli_ops = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class OperatorBasis:
    def __init__(self, basis_ops, labels):
        if len(basis_ops.shape) != 3:
            raise ValueError(
                "Expected 3-dimensional array of operators, "
                "instead got an {}-dimensional one".format(basis_ops.shape)
            )

        if basis_ops.shape[1] != basis_ops.shape[2]:
            raise ValueError(
                "basis operators expected to square matrices, "
                "instead got {}".format(basis_ops.shape[1:])
            )

        dim_hilbert = basis_ops.shape[1]
        if basis_ops.shape[0] > dim_hilbert ** 2:
            raise ValueError(
                "Unexpected number of operators {} given a hilbert "
                "dimensionality of {}".format(basis_ops.shape[0], dim_hilbert)
            )

        self.ops = basis_ops
        self.labels = labels

    def tensor(self, basis):
        tensor_prod_ops = np.kron(self.ops, basis.ops)
        joint_labels = [
            "".join(labels) for labels in product(self.labels, basis.labels)
        ]
        return OperatorBasis(tensor_prod_ops, joint_labels)

    def __matmul__(self, other_basis):
        return self.tensor(other_basis)

    @property
    def dim_hilbert(self):
        return self.ops.shape[1]


def pauli_basis():
    labels = list(pauli_ops.keys())
    ops = np.array(list(pauli_ops.values()))
    basis = OperatorBasis(ops, labels)
    return basis



    






