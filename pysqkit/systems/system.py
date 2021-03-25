from typing import Optional, Dict, Union, Iterable, List

import numpy as np

from ..qubits import Qubit
from ..couplers import CouplingTerm

from ..operators import id_op
from ..util.linalg import tensor_prod


class System:
    def __init__(
        self,
        qubits: Iterable[Qubit],
        couplings: Iterable[CouplingTerm]
    ) -> None:
        qubit_labels = []

        for qubit in qubits:
            if not isinstance(qubit, Qubit):
                raise ValueError(
                    "Each qubit must a pysqkit.Qubit object, "
                    "instead got {}".format(type(qubit)))

            if qubit.label in qubit_labels:
                raise ValueError(
                    "Multiple qubits share the same label, "
                    "please ensure each qubit is uniquely labeled."
                )
            qubit_labels.append(qubit.label)

        self._qubits = list(qubits)
        self._labels = qubit_labels

        for coupling in couplings:
            if not isinstance(coupling, CouplingTerm):
                raise ValueError(
                    "Each coupling should be "
                    "a pysqkit.CouplingTerm "
                    " object, instead got {}".format(type(coupling))
                )
        self._couplings = couplings

    def __len__(self) -> int:
        return len(self._qubits)

    def __getitem__(self, q_label: Union[str, int]) -> Qubit:
        if isinstance(q_label, str):
            try:
                return self._qubits[self._labels.index(q_label)]
            except KeyError:
                raise KeyError(
                    "Qubit {} is not part of the state".format(q_label))
        elif isinstance(q_label, int):
            if q_label >= len(self._qubits) or q_label < 0:
                raise ValueError("Index outside of number of qubits in system")
            return self._qubits[q_label]

    def __contains__(self, qubit: Union[str, Qubit]) -> bool:
        if isinstance(qubit, str):
            return qubit in self._labels
        elif isinstance(qubit, Qubit):
            return qubit in self._qubits
        else:
            raise ValueError(
                "Can't check if a {} object is in system".format(type(qubit)))

    def __iter__(self) -> Iterable[Qubit]:
        yield from self._qubits

    @property
    def size(self):
        return len(self._qubits)

    @property
    def qubits(self) -> List[Qubit]:
        return self._qubits

    def index(self, qubit: Union[str, Qubit]) -> int:
        if isinstance(qubit, str):
            label = qubit
        elif isinstance(qubit, Qubit):
            label = qubit.label
        else:
            raise ValueError(
                "Provided qubit type must be a str or pysqkit.Qubit"
                "instead got {}".format(type(qubit))
            )
        if label in self._labels:
            return self._labels.index(label)
        else:
            raise ValueError("Qubit {} not in system".format(label))

    def bare_hamiltonian(
        self,
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        if truncated_levels:
            subsys_dims = {
                q.label: truncated_levels[q.label] if q in truncated_levels else 6 for q in self.qubits}
        else:
            subsys_dims = {qubit.label: 6 for qubit in self.qubits}

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

        return bare_hamiltonian

    def int_hamiltonian(
        self,
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        if truncated_levels:
            subsys_dims = {
                q.label: truncated_levels[q.label] if q in truncated_levels else 6 for q in self.qubits}
        else:
            subsys_dims = {qubit.label: 6 for qubit in self.qubits}

        sys_dim = np.prod(list(subsys_dims.values()))

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

        return int_hamiltonian

    def hamiltonian(
            self,
            *,
            truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:

        bare_hamiltonian = self.bare_hamiltonian(
            truncated_levels=truncated_levels
        )

        int_hamiltonian = self.int_hamiltonian(
            truncated_levels=truncated_levels
        )

        return bare_hamiltonian + int_hamiltonian
