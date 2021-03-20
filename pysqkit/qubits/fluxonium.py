from typing import Union, Optional
import warnings

import numpy as np
from scipy import linalg as la
from scipy import special as ss

from .qubit import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis
from ..util.linalg import get_mat_elem

_supported_bases = (FockBasis, )

pi = np.pi


class Fluxonium(Qubit):
    def __init__(
        self,
        charge_energy: float,
        induct_energy: float,
        joseph_energy: float,
        flux: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 10
    ) -> None:
        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._flux = flux

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
    def eff_mass(self) -> float:
        return 1/(8*self._ec)

    @property
    def flux_zpf(self) -> float:
        return (2*self._ec/self._el)**0.25

    @property
    def charge_zpf(self) -> float:
        return (self._el/(32*self._ec))**0.25

    def _get_hamiltonian(self) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            osc_hamil = self.res_freq * self.basis.num_op

            flux_phase = np.exp(1j*2*pi*self.flux)
            phase_op = self.basis.flux_op(self.flux_zpf)

            exp_mat = flux_phase * la.expm(1j*phase_op)
            cos_mat = 0.5 * (exp_mat + exp_mat.conj().T)

            hamil = osc_hamil - self.joseph_energy*cos_mat
        else:
            raise NotImplementedError

        return hamil.real

    def hamiltonian(self) -> np.ndarray:
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
        state_ind: Optional[Union[int, np.ndarray]] = 0,
        *,
        truncation_ind: int = None
    ) -> np.ndarray:
        if isinstance(state_ind, int):
            state_ind = np.array([state_ind])

        _, eig_vecs = self.eig_states()

        if truncation_ind is not None:
            if not isinstance(truncation_ind, int) or truncation_ind < 0:
                raise ValueError(
                    "The truncation ind must be a positive integer")

            if truncation_ind > self.dim_hilbert:
                warnings.warn(
                    "Truncation index exceed the Hilber dimension of the basis")
                truncation_ind = self.dim_hilbert
            eig_vecs = eig_vecs[state_ind, :truncation_ind]
            num_inds = truncation_ind

        else:
            eig_vecs = eig_vecs[state_ind]
            num_inds = self.dim_hilbert

        inds = np.arange(num_inds)
        prefactor = (self.eff_mass*self.res_freq/np.pi)**0.25

        nat_phase = phase * np.sqrt(self.eff_mass*self.res_freq)

        coeffs = prefactor * np.einsum(
            'i, b, ai -> aib',
            1/np.sqrt(2.0**inds * ss.factorial(inds)),
            np.exp(-0.5*nat_phase*nat_phase),
            eig_vecs,
            optimize=True
        )

        wave_funcs = np.squeeze([np.polynomial.hermite.hermval(
            nat_phase, coeff, tensor=False) for coeff in coeffs])

        return wave_funcs

    def mat_elements(
        self,
        operator: Union[str, np.ndarray],
        in_states: Optional[np.ndarray] = None,
        out_states: Optional[np.ndarray] = None,
        *,
        in_state_inds: Union[int, np.ndarray] = 10,
        out_state_inds: Union[int, np.ndarray] = 10
    ) -> Union[float, np.ndarray]:

        if isinstance(operator, str):
            if hasattr(self.basis, operator):
                _op = getattr(self.basis, operator)
                op = _op() if callable(_op) else _op
            else:
                raise ValueError(
                    "Given operator string not supported by the basis {}".format(str(self.basis)))
        elif isinstance(operator, np.ndarray):
            op = operator
        else:
            raise ValueError("Incorrect operator provided")

        if in_states is None:
            _, in_states = self.eig_states()
        else:
            if not isinstance(in_states, np.ndarray):
                raise ValueError("Input states format not supported")

        mat_elems = get_mat_elem(op, in_states, out_states)

        return mat_elems
