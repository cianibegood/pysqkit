from typing import Union

import numpy as np
from scipy.linalg import cosm

from .qubit import Qubit


class Fluxonium(Qubit):
    def __init__(
        self,
        charge_energy: float,
        induct_energy: float,
        joseph_energy: float,
        ext_flux: float,
        *,
        dim_hilbert=10
    ) -> None:
        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._ext_flux = ext_flux

        self._dim_hilbert = dim_hilbert

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @dim_hilbert.setter
    def dim_hilbert(self, new_dim: int) -> None:
        self._dim_hilbert = new_dim

    @property
    def charge_energy(self) -> float:
        return self._ec

    @charge_energy.setter
    def charge_energy(self, new_energy: float) -> None:
        self._ec = new_energy

    @property
    def induct_energy(self) -> float:
        return self._el

    @induct_energy.setter
    def induct_energy(self, new_energy: float) -> None:
        self._el = new_energy

    @property
    def joseph_energy(self) -> float:
        return self._ej

    @joseph_energy.setter
    def joseph_energy(self, new_value: float) -> None:
        self._ej = new_value

    @property
    def res_freq(self) -> float:
        return np.sqrt(8*self._ec*self._el)

    @property
    def eff_mass(self) -> float:
        return 1/(8*self._ec)

    @property
    def flux_zpf(self) -> float:
        return (2*self._ec/self._el)**(1/4)

    def hamiltonian(self) -> np.ndarray:
        low_op = np.diag(np.sqrt(np.arange(1, self._dim_hilbert)), 1)
        id_op = np.eye(self._dim_hilbert)

        hamil = self.res_freq*(low_op.conj().T @ low_op + id_op/2)
        total_flux = self.flux_zpf*(low_op + low_op.conj().T) + self._ext_flux
        hamil -= self._ej*cosm(total_flux)
        return hamil

    def potential(self, flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pot = self._el/2*flux**2 - self._ej*np.cos(flux + self._ext_flux)
        return pot

    def wave_function(self):
        return None
