from typing import Union, Optional, List, Tuple
import warnings
from copy import copy

import numpy as np
from scipy import linalg as la
from scipy import special as ss
import xarray as xr
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis
from ..util.linalg import get_mat_elem
from ..util.phys import average_photon


_supported_bases = (FockBasis,)

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
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._flux = flux
        self._scq_compatible = False  # Used for debugging and comparison

        if basis is None:
            # try-catch block here in case dim_hilbert is wrong
            basis = fock_basis(dim_hilbert)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)

    def __copy__(self) -> "Fluxonium":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy,
            self.induct_energy,
            self.joseph_energy,
            self.flux,
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
        return np.sqrt(8 * self._ec * self._el)

    @property
    def osc_mass(self) -> float:
        return 1 / (8 * self._ec)

    @property
    def osc_len(self) -> float:
        return (8 * self._ec / self._el) ** 0.25

    @property
    def flux_zpf(self) -> float:
        return (2 * self._ec / self._el) ** 0.25

    @property
    def charge_zpf(self) -> float:
        return (self._el / (32 * self._ec)) ** 0.25

    def _get_charge_op(self):
        charge_op = 1j * self.charge_zpf * (self.basis.raise_op - self.basis.low_op)

        return charge_op

    def charge_op(
        self,
        *,
        as_qobj=False,
    ) -> np.ndarray:
        charge_op = self.basis.finalize_op(self._get_charge_op())

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

    def flux_op(self, *, as_qobj=False) -> np.ndarray:
        flux_op = self.basis.finalize_op(self._get_flux_op())

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
                osc_hamil = self.res_freq * (self.basis.num_op + 0.5 * self.basis.id_op)

            flux_phase = np.exp(1j * 2 * pi * self.flux)

            exp_mat = flux_phase * la.expm(1j * self._get_flux_op())
            cos_mat = 0.5 * (exp_mat + exp_mat.conj().T)

            hamil = osc_hamil - self.joseph_energy * cos_mat
        else:
            raise NotImplementedError

        return hamil.real

    def hamiltonian(self, *, as_qobj=False) -> np.ndarray:
        hamil = self.basis.finalize_op(self._get_hamiltonian())

        if as_qobj:
            dim = self.basis.sys_truncated_dims

            qobj_op = Qobj(
                inpt=hamil, dims=[dim, dim], shape=hamil.shape, type="oper", isherm=True
            )
            return qobj_op
        return hamil

    def potential(self, flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pot = 0.5 * self._el * flux * flux - self._ej * np.cos(
            flux + 2 * np.pi * self.flux
        )
        return pot

    def wave_function(
        self,
        phase: Union[float, np.ndarray],
        levels: Optional[Union[int, np.ndarray]] = 0,
        *,
        truncation_ind: int = None,
        get_data=False,
    ) -> np.ndarray:
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

    def dielectric_rates(
        self, level_k: int, level_m: int, qdiel: float, beta: float, return_op=False
    ) -> Tuple[float, float]:
        if qdiel < 0 or beta < 0:
            raise ValueError(
                "Quality factor qdiel and (absolute) "
                "inverse temperature beta must be positive."
            )

        if level_k == level_m:
            raise ValueError(
                "Eigenstate labels level_k and level_m " "must be different."
            )

        if level_k < 0 or level_m < 0:
            raise ValueError(
                "Eigenstate labels level_k and level_m must " "be positive."
            )
        elif level_k >= self.dim_hilbert or level_m >= self.dim_hilbert:
            raise ValueError(
                "Eigenstate labels k and m must be smaller than"
                " the Hilbert space dimension."
            )

        if level_k > level_m:
            level_k, level_m = level_m, level_k

        eig_en, eig_vec = self.eig_states([level_k, level_m])
        energy_diff = (eig_en[1] - eig_en[0]) / self._ec
        phi_km = get_mat_elem(self.flux_op(), eig_vec[1], eig_vec[0])

        gamma = self._ec * 1 / (4 * qdiel) * energy_diff ** 2 * np.abs(phi_km) ** 2
        nth = average_photon(energy_diff * self._ec, beta)

        relaxation_rate = gamma * (nth + 1)
        excitation_rate = gamma * nth

        if return_op:
            down_op = np.outer(eig_vec[0], np.conj(eig_vec[1]))
            return relaxation_rate, excitation_rate, down_op

        return relaxation_rate, excitation_rate

    def _get_dielectric_jump(
        self, level_k: int, level_m: int, qdiel: float, beta: float
    ) -> Tuple[np.ndarray]:

        relaxation_rate, excitation_rate, down_op = self.dielectric_rates(
            level_k, level_m, qdiel, beta, return_op=True
        )

        up_op = down_op.conj().T

        jump_down = np.sqrt(relaxation_rate) * down_op
        jump_up = np.sqrt(excitation_rate) * up_op

        return jump_down, jump_up

    def dielectric_jump(
        self, level_k: int, level_m: int, qdiel: float, beta: float, as_qobj=False
    ) -> Tuple[np.ndarray]:
        jump_down, jump_up = self._get_dielectric_jump(level_k, level_m, qdiel, beta)
        if as_qobj:
            dim = self.dim_hilbert
            jump_down_qobj = Qobj(
                inpt=jump_down,
                dims=[[dim], [dim]],
                shape=[dim, dim],
                type="oper",
                isherm=True,
            )
            jump_up_qobj = Qobj(
                inpt=jump_up,
                dims=[[dim], [dim]],
                shape=[dim, dim],
                type="oper",
                isherm=True,
            )
            return jump_down_qobj, jump_up_qobj
        return jump_down, jump_up

    def dielectric_loss(
        self, qdiel: float, beta: float, as_qobj=False
    ) -> List[np.ndarray]:
        jump_list = []
        for level_k in range(0, self.dim_hilbert):
            for level_m in range(level_k + 1, self.dim_hilbert):
                jump_list.extend(
                    self.dielectric_jump(level_k, level_m, qdiel, beta, as_qobj)
                )
        return jump_list
