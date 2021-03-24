from typing import Optional, List, Dict

import numpy as np

from ..util.linalg import tensor_prod
from ..qubits import Qubit

from ..operators import id_op


class System:
    def __init__(self, qubits: Qubit, couplings):
        self._qubits = qubits
        self._couplings = couplings

    def __len__(self) -> int:
        return len(self._qubits)

    def __getitem__(self, key):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __index__(self) -> int:
        raise NotImplementedError

    @property
    def size(self):
        return len(self._qubits)

    @property
    def qubits(self):
        return self._qubits

    def hamiltonian(
            self,
            *,
            bare=False,
            truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        subsys_dims = {}
        if truncated_levels:
            for qubit in self.qubits:
                q_label = qubit.label
                if q_label in truncated_levels:
                    subsys_dims[q_label] = truncated_levels[q_label]
                else:
                    subsys_dims[q_label] = 6
        else:
            for qubit in self.qubits:
                subsys_dims[qubit.label] = 6

        sys_dim = np.prod(list(subsys_dims.values()))

        bare_hamiltonian = np.zeros((sys_dim, sys_dim), dtype=complex)

        for subsys_ind, subsys_qubit in enumerate(self.qubits):
            eig_energies = subsys_qubit.eig_energies(
                levels=subsys_dims[subsys_qubit.label])
            q_hamil = np.diag(eig_energies)

            subsys_ops = [q_hamil if q.label == subsys_qubit.label else id_op(
                subsys_dims[q.label]) for q in self.qubits]
            subsys_hamil = tensor_prod(subsys_ops)

            bare_hamiltonian += subsys_hamil

        if bare:
            return bare_hamiltonian

        int_hamiltonian = np.zeros((sys_dim, sys_dim), dtype=complex)

        for coup in self._couplings:
            for prefactor, term_ops in coup.hamiltonian_terms(tensor_ops=False):
                coupled_qubits = list(term_ops.keys())

                diag_ops = []
                for qubit in self.qubits:
                    if qubit.label in coupled_qubits:
                        q_op = qubit.mat_elements(
                            term_ops[qubit.label],
                            levels=subsys_dims[qubit.label]
                        )
                        diag_ops.append(q_op)
                    else:
                        diag_ops.append(id_op(subsys_dims[qubit.label]))

                op = tensor_prod(diag_ops)

                int_hamiltonian += prefactor * op

        return bare_hamiltonian + int_hamiltonian
