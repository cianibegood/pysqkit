from abc import ABC, abstractmethod
from typing import Tuple, Optional, Iterable, Union

import numpy as np
from scipy import linalg as la
import xarray as xr
from qutip import Qobj

from ..util.linalg import order_vecs, get_mat_elem


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

    @abstractmethod
    def _qubit_attrs(self) -> dict:
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

    def mat_elements(
        self,
        operator: Union[str, np.ndarray],
        in_states: Optional[np.ndarray] = None,
        out_states: Optional[np.ndarray] = None,
        levels: Union[int, np.ndarray] = 10,
        *,
        get_data=False,
    ) -> np.ndarray:
        if isinstance(operator, str):
            if hasattr(self.basis, operator):
                _op = getattr(self.basis, operator)
                op = _op() if callable(_op) else _op
                if not isinstance(op, np.ndarray):
                    raise ValueError("Obtained operator is not a numpy array")
                if len(op.shape) != 2:
                    raise ValueError("The operator must be a 2D array")

            else:
                raise ValueError(
                    "Given operator string not supported by the basis {}".format(str(self.basis)))
        elif isinstance(operator, np.ndarray):
            op = operator
        else:
            raise ValueError("Incorrect operator provided")

        if in_states is None:
            _, in_states = self.eig_states(levels=levels)
        else:
            raise NotImplementedError

        if out_states is None:
            _, out_states = self.eig_states(levels=levels)
        else:
            raise NotImplementedError

        mat_elems = get_mat_elem(op, in_states, out_states)

        if get_data:
            return mat_elems

        data_arr = xr.DataArray(
            data=mat_elems,
            dims=['in_leves', 'out_levels'],
            coords=dict(
                in_levels=levels,
                out_levens=levels,
            ),
            attrs=dict(
                operator=op,
                dim_hilbert=self.dim_hilbert,
                basis=str(self.basis),
                **self._qubit_attrs
            )
        )

        return data_arr

        def diag_hamiltonian(
                self,
                levels: Optional[Union[int, Iterable[int]]] = None
        ) -> np.ndarray:
            eig_energies = self.eig_energies(levels=levels)

            hamil = np.diag(eig_energies)
            return hamil
