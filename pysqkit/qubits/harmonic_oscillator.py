from typing import Union, Optional, List
import warnings
from copy import copy

import numpy as np
# from scipy import linalg as la
from scipy import special as ss
import xarray as xr
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis


_supported_bases = (FockBasis,)


class HarmonicOscillator(Qubit):
    
    """
    Description
    --------------------------------------------------------------------------
    Class associated with a harmonic oscillator with Hamiltonian

    H = 4 E_C q^2 + E_L \phi^2/2

    with E_C the charging energy and E_L the inductive energy.
    q is the dimensionless charge and \phi the dimensionless flux and in 
    the quantum setting they satisfy commutation relation [\phi, q] = i
    """

    def __init__(
        self,
        label: str,
        charge_energy: float,
        induct_energy: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        
        """
        Parameters
        ----------------------------------------------------------------------
        label: str
            A string that identifies the harmonic oscillator
        charge_energy: float
            Charging energy of the harmonic oscillator
        induct_energy: float
            Inductive energy of the harmonic oscillator
        basis: Optional[OperatorBasis] = None
            Basis in which we want to write the operators. If not provided
            it is assumed it is the Fock basis
        dim_hilbert: Optional[int] = 100
            Hilber space dimension       
        """

        self._ec = charge_energy
        self._el = induct_energy

        if basis is None:
            basis = fock_basis(dim_hilbert)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)

    def __copy__(self) -> "HarmonicOscillator":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy,
            self.induct_energy,
            basis=copy(self.basis),
        )
        qubit_copy._drives = {
            label: copy(drive) for label, drive in self._drives.items()
        }
        return qubit_copy

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

    def _get_charge_op(self):
        charge_op = 1j*self.charge_zpf*(self.basis.raise_op - self.basis.low_op)
        return charge_op

    def charge_op(
        self,
        *,
        expand=True,
        as_qobj=False,
    ) -> np.ndarray:
        op = self._get_charge_op()
        charge_op = self.basis.finalize_op(op, expand=expand)

        if as_qobj:
            dim = self.basis.sys_truncated_dims

            qobj_op = Qobj(
                inpt=charge_op,
                dims=[dim, dim],
                shape=charge_op.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op
        return charge_op

    def _get_flux_op(self):
        flux_op = self.flux_zpf * (self.basis.raise_op + self.basis.low_op)
        return flux_op

    def flux_op(self, *, expand=True, as_qobj=False) -> np.ndarray:
        op = self._get_flux_op()
        flux_op = self.basis.finalize_op(op, expand=expand)

        if as_qobj:
            dim = self.basis.sys_truncated_dims

            qobj_op = Qobj(
                inpt=flux_op,
                dims=[dim, dim],
                shape=flux_op.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op
        return flux_op

    @property
    def _qubit_attrs(self) -> dict:
        q_attrs = dict(
            charge_energy=self.charge_energy,
            induct_energy=self.induct_energy,
        )
        return q_attrs

    def _get_hamiltonian(
        self
    ) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            osc_hamil = self.res_freq*(self.basis.num_op + \
                 0.5*self.basis.id_op)
        else:
            raise NotImplementedError

        return osc_hamil.real

    def hamiltonian(self, *, expand=True, as_qobj=False) -> np.ndarray:
        op = self._get_hamiltonian()
        hamil = self.basis.finalize_op(op, expand=expand)

        if as_qobj:
            dim = self.basis.sys_truncated_dims
            qobj_op = Qobj(
                inpt=hamil,
                dims=[dim, dim],
                shape=hamil.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op
        return hamil

    def potential(self, flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pot = 0.5 * self._el * flux * flux
        return pot

    def wave_function(
        self,
        phase: Union[float, np.ndarray],
        levels: Optional[Union[int, np.ndarray]] = 0,
        *,
        truncation_ind: int = None,
        get_data=False,
    ) -> xr.Dataset:
        if isinstance(levels, int):
            levels = np.array([levels])

        if isinstance(phase, float):
            phase = np.array([phase])

        eig_vals, eig_vecs = self.eig_states(levels=levels)

        if truncation_ind is not None:
            if not isinstance(truncation_ind, int) or truncation_ind < 0:
                raise ValueError("The truncation ind must be a positive integer")

            if truncation_ind > self.dim_hilbert:
                warnings.warn(
                    "Truncation index exceed the Hilber dimension of the basis"
                )
                truncation_ind = self.dim_hilbert
            eig_vecs = eig_vecs[:, :truncation_ind]

        inds = np.arange(truncation_ind or self.dim_hilbert)
        prefactor = (self.osc_mass * self.res_freq / np.pi) ** 0.25

        nat_phase = phase * np.sqrt(self.osc_mass * self.res_freq)

        coeffs = prefactor * np.einsum(
            "i, b, ai -> aib",
            1 / np.sqrt(2.0 ** inds * ss.factorial(inds)),
            np.exp(-0.5 * nat_phase * nat_phase),
            eig_vecs,
            optimize=True,
        )

        wave_funcs = np.array(
            [
                np.polynomial.hermite.hermval(nat_phase, coeff, tensor=False)
                for coeff in coeffs
            ]
        )

        if get_data:
            return wave_funcs

        dataset = xr.Dataset(
            data_vars=dict(
                wave_func=(["level", "phase"], wave_funcs),
                energy=(["level"], eig_vals),
                potential=(["phase"], self.potential(phase)),
            ),
            coords=dict(
                level=levels,
                phase=phase,
            ),
            attrs=dict(
                dim_hilbert=self.dim_hilbert,
                truncation_ind=truncation_ind,
                basis=str(self.basis),
                **self._qubit_attrs,
            ),
        )

        return dataset

    def collapse_ops(
        self
    ):
        pass 
    
    def dephasing_op(
        self
    ):
        pass
        

# def _get_trans_ops(in_state, out_state):
#     down_op = np.outer(in_state, out_state.conj())
#     up_op = np.transpose(down_op.conj())
#     return down_op, up_op
