from typing import Union, Iterable, Tuple, Dict, Optional, List
from itertools import chain

import numpy as np

from ..util.linalg import tensor_prod


class CouplingTerm:
    def __init__(
            self,
            prefactors: Union[float, complex, Iterable[Union[float, complex]]],
            operators: Iterable[Dict[str, np.ndarray]],
            qubits: Optional[List[str]] = None
    ):

        if isinstance(operators, dict):
            self._ops = [operators]
        elif isinstance(operators, Iterable):
            self._ops = operators

        for op in self._ops:
            for key, val in op.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "Operator keys must be qubit labels of type str")
                if not isinstance(val, np.ndarray):
                    raise ValueError("The operators must be np.ndarray type")

        if qubits is None:
            qubit_set = set(chain.from_iterable(op.keys() for op in self._ops))
            self._qubits = sorted(list(qubit_set))
        else:
            involved_qubits = set(chain.from_iterable(op.keys()
                                                      for op in self._ops))

            qubit_set = set(qubits)
            if len(qubit_set) != len(qubits):
                raise ValueError("There are duplicate labels in the qubits")
            if qubit_set != involved_qubits:
                raise ValueError(
                    "Qubit labels have a mismatch with the operator labels")
            self._qubits = qubits

        self._hilbert_dims = {}
        for op in self._ops:
            for qubit, qubit_op in operators.items():
                if len(qubit_op.shape) != 2 or qubit_op.shape[0] != qubit_op.shape[1]:
                    raise ValueError("Each operator must be a square matrix")

                if qubit in self._hilbert_dims:
                    if self._hilbert_dims[qubit] != qubit_op.shape[0]:
                        raise ValueError(
                            "Mismatch in the dimensionality of the qubit operators in different terms")
                else:
                    self._hilbert_dims[qubit] = qubit_op.shape[0]

        if isinstance(prefactors, (float, int, complex)):
            self._prefactors = [prefactors] * len(self._ops)
        elif isinstance(prefactors, Iterable):
            if len(prefactors) != len(self._ops):
                raise ValueError(
                    "Number of provided operators does not correspond to the number of terms in the Hamiltonian")
            self._prefactors = list(prefactors)

    @property
    def qubits(self):
        return self._qubits

    @property
    def hilbert_dims(self):
        return self._hilbert_dims

    @property
    def hilbert_dim(self):
        return np.prod(list(self._hilbert_dims.values()))

    def _get_hamiltonian(self) -> np.ndarray:
        dim = self.hilbert_dim
        hamiltonian = np.zeros((dim, dim), dtype='complex')

        for prefactor, op in self.hamiltonian_terms():
            hamiltonian += prefactor * op

        if np.all(np.real(hamiltonian)):
            return hamiltonian.real
        return hamiltonian

    def hamiltonian(self) -> np.ndarray:
        hamil = self._get_hamiltonian()
        return hamil

    def hamiltonian_terms(
            self,
            *,
            tensor_ops=True
    ) -> Iterable[Tuple[Union[float, complex], Union[np.ndarray, Dict[str, np.ndarray]]]]:
        if tensor_ops:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                inv_qs = list(term_ops.keys())
                total_ops = list(
                    term_ops[q] if q in inv_qs else np.eye(self.hilbert_dims[q]) for q in self._qubits
                )
                yield prefactor, tensor_prod(total_ops)
        else:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                yield prefactor, term_ops
