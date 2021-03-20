from abc import ABC, abstractmethod
from typing import Tuple, Optional, Iterable, Union

import numpy as np
from scipy import linalg as la
from qutip import Qobj

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

    def _get_eig_vals(self, subset_inds: Tuple[int]) -> np.ndarray:
        hamil = self.hamiltonian()
        if isinstance(hamil, Qobj):
            return hamil.eigenenergies()
        eig_vals = la.eigh(
            hamil,
            eigvals_only=True,
            subset_by_index=subset_inds
        )
        return eig_vals

    def _get_eig_states(self, subset_inds: Tuple[int]) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.hamiltonian()
        if isinstance(hamil, Qobj):
            return hamil.eigenstates()
        eig_vals, eig_vecs = la.eigh(
            hamil,
            eigvals_only=False,
            subset_by_index=subset_inds
        )
        return eig_vals, eig_vecs.T

    def eig_energies(
        self,
        levels: Optional[Union[int, Iterable[int]]] = None
    ) -> np.ndarray:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals = self._get_eig_vals(subset_inds)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds])
        return order_vecs(eig_vals)

    def eig_states(
            self,
            levels: Optional[Union[int, Iterable[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals, eig_vecs = self._get_eig_states(subset_inds)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds], eig_vecs[sel_inds])
        return order_vecs(eig_vals, eig_vecs)
