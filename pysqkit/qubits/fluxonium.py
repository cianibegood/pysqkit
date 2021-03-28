from typing import Union, Optional
import warnings

import numpy as np
from scipy import linalg as la
from scipy import special as ss
import xarray as xr

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis

_supported_bases = (FockBasis, )

pi = np.pi


class Fluxonium(Qubit):
    def __init__(
        self,
        label: str,
        charge_energy: float,
        induct_energy: float,
        joseph_energy: float,
        flux: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100
    ) -> None:
        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._flux = flux
        self._scq_compatible = False  # Used for debugging and comparison

        if basis is None:
            # try-catch block here in case dim_hilbert is wrong
            basis = fock_basis(dim_hilbert, self.osc_len)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)

    @property
    def charge_energy(self) -> float:
        return self._ec

    @charge_energy.setter
    def charge_energy(self, charge_energy: float) -> None:
        self._ec = charge_energy

    @property
    def induct_energy(self) -> float:
        return self._el

    @induct_energy.setter
    def induct_energy(self, induct_energy: float) -> None:
        self._el = induct_energy

    @property
    def joseph_energy(self) -> float:
        return self._ej

    @joseph_energy.setter
    def joseph_energy(self, joseph_energy: float) -> None:
        self._ej = joseph_energy

    @property
    def flux(self) -> float:
        return self._flux

    @flux.setter
    def flux(self, flux: float) -> None:
        self._flux = flux

    @property
    def res_freq(self) -> float:
        return np.sqrt(8*self._ec*self._el)

    @property
    def osc_mass(self) -> float:
        return 1/(8*self._ec)

    @property
    def osc_len(self) -> float:
        return (8*self._ec/self._el)**0.25

    @property
    def flux_zpf(self) -> float:
        return (2*self._ec/self._el)**0.25

    @property
    def charge_zpf(self) -> float:
        return (self._el/(32*self._ec))**0.25

    @property
    def _qubit_attrs(self) -> dict:
        q_attrs = dict(
            charge_energy=self.charge_energy,
            induct_energy=self.induct_energy,
            joseph_energy=self.joseph_energy,
            flux=self.flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            if self._scq_compatible:
                osc_hamil = self.res_freq * self.basis.num_op
            else:
                osc_hamil = self.res_freq * \
                    (self.basis.num_op + 0.5*self.basis.id_op)

            flux_phase = np.exp(1j*2*pi*self.flux)

            exp_mat = flux_phase * la.expm(1j*self.basis.flux_op)
            cos_mat = 0.5 * (exp_mat + exp_mat.conj().T)

            hamil = osc_hamil - self.joseph_energy*cos_mat
        else:
            raise NotImplementedError

        return hamil.real

    def hamiltonian(
        self,
    ) -> np.ndarray:
        hamil = self._get_hamiltonian()
        return hamil

    def potential(
        self,
        flux: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pot = 0.5*self._el*flux*flux - self._ej * \
            np.cos(flux + 2*np.pi*self.flux)
        return pot

    def wave_function(
        self,
        phase: Union[float, np.ndarray],
        levels: Optional[Union[int, np.ndarray]] = 0,
        *,
        truncation_ind: int = None,
        get_data=False
    ) -> np.ndarray:
        if isinstance(levels, int):
            levels = np.array([levels])

        if isinstance(phase, float):
            phase = np.array([phase])

        eig_vals, eig_vecs = self.eig_states(levels=levels)

        if truncation_ind is not None:
            if not isinstance(truncation_ind, int) or truncation_ind < 0:
                raise ValueError(
                    "The truncation ind must be a positive integer")

            if truncation_ind > self.dim_hilbert:
                warnings.warn(
                    "Truncation index exceed the Hilber dimension of the basis")
                truncation_ind = self.dim_hilbert
            eig_vecs = eig_vecs[:, :truncation_ind]

        inds = np.arange(truncation_ind or self.dim_hilbert)
        prefactor = (self.osc_mass*self.res_freq/np.pi)**0.25

        nat_phase = phase * np.sqrt(self.osc_mass*self.res_freq)

        coeffs = prefactor * np.einsum(
            'i, b, ai -> aib',
            1/np.sqrt(2.0**inds * ss.factorial(inds)),
            np.exp(-0.5*nat_phase*nat_phase),
            eig_vecs,
            optimize=True
        )

        wave_funcs = np.array([np.polynomial.hermite.hermval(
            nat_phase, coeff, tensor=False) for coeff in coeffs])

        if get_data:
            return wave_funcs

        dataset = xr.Dataset(
            data_vars=dict(
                wave_func=(['level', 'phase'], wave_funcs),
                energy=(['level'], eig_vals),
                potential=(['phase'], self.potential(phase))
            ),
            coords=dict(
                level=levels,
                phase=phase,
            ),
            attrs=dict(
                dim_hilbert=self.dim_hilbert,
                truncation_ind=truncation_ind,
                basis=str(self.basis),
                **self._qubit_attrs
            )
        )

        return dataset
