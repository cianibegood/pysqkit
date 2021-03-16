from typing import Union, Optional
import warnings

import numpy as np
from scipy.linalg import cosm
from scipy.special import factorial

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
        return 0.125*self._ec

    @property
    def flux_zpf(self) -> float:
        return (2*self._ec/self._el)**0.25

    @property
    def charge_zpf(self) -> float:
        return (self._el/(32*self._ec))**0.25

    def _get_hamiltonian(self) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            osc_hamil = self.res_freq * \
                (self._basis.num_op + 0.5*self._basis.id_op)
            total_flux = self._basis.flux_op(self.flux_zpf) + self._ext_flux
            cos_term = cosm(total_flux)
            hamil = osc_hamil - self._ej*cos_term
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
        pot = 0.5*self._el*(flux**2) - self._ej*np.cos(flux + self._ext_flux)
        return pot

    def wave_function(
        self,
        flux: Union[float, np.ndarray],
        level_ind: Optional[Union[int, np.ndarray]] = 0,
        *,
        truncation_ind: int = None
    ) -> np.ndarray:
        if isinstance(level_ind, int):
            level_ind = np.array([level_ind])

        _, eig_vecs = self.eig_states()

        if truncation_ind is not None:
            if not isinstance(truncation_ind, int) or truncation_ind < 0:
                raise ValueError(
                    "The truncation ind must be a positive integer")

            if truncation_ind > self.dim_hilbert:
                warnings.warn(
                    "Truncation index exceed the Hilber dimension of the basis")
                truncation_ind = self.dim_hilbert
            eig_vecs = eig_vecs[level_ind, :truncation_ind]
            num_inds = truncation_ind

        else:
            eig_vecs = eig_vecs[level_ind]
            num_inds = self.dim_hilbert

        inds = np.arange(num_inds)
        _c = (self.eff_mass*self.res_freq/np.pi)**0.25

        coeffs = _c * np.einsum(
            'i, b, ai -> abi',
            1/np.sqrt(np.power(2.0, inds) * factorial(inds)),
            np.exp(-0.5*flux**2),
            eig_vecs
        )

        wave_funcs = np.squeeze([np.polynomial.hermite.hermval(
            flux, coeff.T, tensor=False) for coeff in coeffs])

        return wave_funcs
