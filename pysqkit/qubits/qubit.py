from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import linalg

from ..util.linalg import order_vecs


class Qubit(ABC):
    def __init__(self, basis):
        self._basis = basis

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, new_basis):
        self._basis = new_basis

    @property
    def dim_hilbert(self):
        return self._basis.dim_hilbert

    @abstractmethod
    def hamiltonian(self) -> np.ndarray:
        pass

    @abstractmethod
    def potential(self) -> np.ndarray:
        pass

    @abstractmethod
    def wave_function(self) -> np.ndarray:
        pass

    def _get_eig_vals(self) -> np.ndarray:
        hamil = self.hamiltonian()
        eig_vals = linalg.eigh(hamil, eigvals_only=True)
        return order_vecs(eig_vals)

    def _get_eig_states(self) -> Tuple[np.ndarray, np.ndarray]:
        # FIXME: This function doesn't return the correct eigen states (vs legacy code)
        hamil = self.hamiltonian()
        eig_vals, eig_vecs = linalg.eigh(hamil, eigvals_only=False)
        return order_vecs(eig_vals, eig_vecs)

    def eig_energies(self) -> np.ndarray:
        eig_vals = self._get_eig_vals()
        return eig_vals

    def eig_states(self) -> Tuple[np.ndarray, np.ndarray]:
        eig_states = self._get_eig_states()
        return eig_states
