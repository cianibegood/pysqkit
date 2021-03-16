from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import linalg

from ..util.linalg import order_vecs


class Qubit(ABC):

    @abstractmethod
    def hamiltonian(self):
        pass

    @abstractmethod
    def potential(self):
        pass

    @abstractmethod
    def wave_function(self):
        pass

    def _get_eig_vals(self) -> np.ndarray:
        hamil = self.hamiltonian()
        eig_vals = linalg.eigh(hamil, eigvals_only=True)
        return order_vecs(eig_vals)

    def _get_eig_states(self) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.hamiltonian()
        eig_vals, eig_vecs = linalg.eigh(hamil, eigvals_only=False)
        return order_vecs(eig_vals, eig_vecs)

    def eig_energies(self) -> np.ndarray:
        eig_vals = self._get_eig_states()
        return eig_vals

    def eig_states(self) -> Tuple[np.ndarray, np.ndarray]:
        eig_states = self._get_eig_states()
        return eig_states
