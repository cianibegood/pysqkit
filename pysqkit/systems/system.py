import numpy as np


from ..util.linalg import tensor_prod
from ..qubits import Qubit


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
            bare=False) -> np.ndarray:
        subsys_hamiltonians = []

        for qi, qubit in enumerate(self.qubits):
            q_hamil = qubit.diag_hamiltonian()
            subsys_hamil = tensor_prod(q_hamil if i == qi else q.basis.id_op
                                       for i, q in enumerate(self.qubits))

            subsys_hamiltonians.append(subsys_hamil)

        bare_hamiltonian = np.sum(subsys_hamiltonians, axis=0)

        if bare:
            return bare_hamiltonian

        int_hamiltonians = []

        for coup in self._couplings:
            coup_hamil = coup.diag_hamiltonian()  # to be implemented still
            subsys_hamil = tensor_prod(coup_hamil if i == qi else q.basis.id_op
                                       for i, q in enumerate(self.qubits))

            int_hamiltonians.append(subsys_hamil)

        int_hamiltonian = np.sum(int_hamiltonians)

        return bare_hamiltonian + int_hamiltonian
