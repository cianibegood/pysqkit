from typing import Union, Optional

import numpy as np
from scipy.linalg import cosm

from .qubit import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis

_supported_bases = (FockBasis, )


class Fluxonium(Qubit):
    def __init__(
        self,
        charge_energy: float,
        induct_energy: float,
        joseph_energy: float,
        ext_flux: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 10
    ) -> None:
        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._ext_flux = ext_flux

        if basis is None:
            # try-catch block here in case dim_hilbert is wrong
            basis = fock_basis(dim_hilbert)
            self._basis = basis
        else:
            if not isinstance(basis, OperatorBasis):
                raise ValueError(
                    "basis must be an instance of bases.OperatorBasis class")
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")
            self._basis = basis

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

    def _get_hamiltonian(self) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            hamil = self.res_freq*(self._basis.raise_op @
                                   self._basis.low_op + self._basis.id_op/2)
            total_flux = self._basis.flux_op(self.flux_zpf) + self._ext_flux

            hamil -= self.joseph_energy*cosm(total_flux)
        else:
            raise NotImplementedError

        return hamil

    def hamiltonian(self) -> np.ndarray:
        hamil = self._get_hamiltonian()
        return hamil

    def potential(
        self,
        flux: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pot = self._el/2*flux**2 - self._ej*np.cos(flux + self._ext_flux)
        return pot

    def wave_function(self):
        return None
