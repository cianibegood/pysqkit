from typing import Optional, Union, List, Tuple
from copy import copy
from warnings import warn

import numpy as np
from scipy import linalg as la
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis
from ..util.linalg import get_mat_elem
from ..util.phys import average_photon

_supported_bases = (FockBasis,)

pi = np.pi


class Transmon(Qubit):
    def __init__(
        self,
        label: str,
        charge_energy: float,
        joseph_energy: float,
        ext_flux: Optional[float] = 0,
        charge_offset: Optional[float] = 0,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        self._ec = charge_energy
        self._ej = joseph_energy
        self._ext_flux = ext_flux
        self._n_offset = charge_offset

        if basis is None:
            basis = fock_basis(dim_hilbert)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)

    def __copy__(self) -> "Transmon":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy,
            self.joseph_energy,
            self.ext_flux,
            self.charge_offset,
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
    def joseph_energy(self) -> float:
        return self._ej

    @joseph_energy.setter
    def joseph_energy(self, joseph_energy: float) -> None:
        self._ej = joseph_energy

    @property
    def ext_flux(self) -> float:
        return self._ext_flux

    @ext_flux.setter
    def ext_flux(self, flux_val: float) -> None:
        self._ext_flux = flux_val

    @property
    def charge_offset(self) -> float:
        return self._n_offset

    @charge_offset.setter
    def charge_offset(self, offset_val: float) -> None:
        self._n_offset = offset_val

    @property
    def res_freq(self) -> float:
        return np.sqrt(8 * self._ec * self._ej)

    @property
    def flux_zpf(self) -> float:
        return (2 * self._ec / self._ej) ** 0.25

    @property
    def charge_zpf(self) -> float:
        return (self._ej / (32 * self._ec)) ** 0.25

    def _get_charge_op(self):
        charge_op = 1j * self.charge_zpf * (self.basis.raise_op - self.basis.low_op)
        return charge_op - self._n_offset

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
            joseph_energy=self.joseph_energy,
            flux=self.flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        charge_op = self._get_charge_op()
        charge_term = 4 * self._ec * (charge_op @ charge_op)

        flux_op = self._get_flux_op()
        joseph_term = self.joseph_energy * la.cosm(flux_op)

        hamil = charge_term - joseph_term
        return hamil

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
        pot = -self._ej * np.cos(flux)
        return pot

    def wave_function(self) -> np.ndarray:
        raise NotADirectoryError

    def dielectric_loss(self) -> List[np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def from_freq(
        label: str,
        max_freq: float,
        anharm: float,
        ext_flux: Optional[float] = 0,
        charge_offset: Optional[float] = 0.5,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> "Transmon":
        charge_energy = -anharm
        res_freq = max_freq - anharm
        joseph_energy = (res_freq / np.sqrt(8 * charge_energy)) ** 2
        return Transmon(
            label,
            charge_energy,
            joseph_energy,
            ext_flux,
            charge_offset,
            basis=basis,
            dim_hilbert=dim_hilbert,
        )


class SimpleTransmon(Qubit):
    def __init__(
        self,
        label: str,
        max_freq: float,
        anharm: float,
        ext_flux: Optional[float] = 0,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        if max_freq <= 0:
            raise ValueError("Frequency expected to be positive value greater than 0.")
        self._freq = max_freq

        if anharm > 0:
            warn(
                "anharm expected to be negative. "
                "Setting as {} instead.".format(-anharm)
            )
            anharm = -anharm
        self._anharm = anharm
        self._ext_flux = ext_flux
        self._ec = np.abs(self._anharm)

        if basis is None:
            basis = fock_basis(dim_hilbert)
        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")
        super().__init__(label=label, basis=basis)

    def __copy__(self) -> "SimpleTransmon":
        qubit_copy = self.__class__(
            self.label,
            self.freq,
            self.anharm,
            self.ext_flux,
            basis=copy(self.basis),
        )
        qubit_copy._drives = {
            label: copy(drive) for label, drive in self._drives.items()
        }
        return qubit_copy

    @property
    def freq(self) -> float:
        res_freq = self._freq - self._anharm
        shifted_freq = res_freq * np.sqrt(np.abs(np.cos(pi * self._ext_flux)))
        return shifted_freq + self._anharm

    @property
    def max_freq(self) -> float:
        return self._freq

    @max_freq.setter
    def max_freq(self, freq_val: float) -> None:
        self._freq = freq_val

    @property
    def anharm(self) -> float:
        return self._anharm

    @anharm.setter
    def anharm(self, anharm_val: float) -> None:
        self._anharm = anharm_val

    @property
    def ext_flux(self) -> float:
        return self._ext_flux

    @ext_flux.setter
    def ext_flux(self, flux: float) -> None:
        self._ext_flux = flux

    @property
    def charge_energy(self) -> float:
        return -self._anharm

    @property
    def joseph_energy(self) -> float:
        res_freq = self._freq - self._anharm
        joseph_energy = (res_freq / np.sqrt(8 * self.charge_energy)) ** 2
        det_joseph_energy = joseph_energy * np.abs(np.cos(pi * self.ext_flux))
        return det_joseph_energy

    @property
    def res_freq(self) -> float:
        return np.sqrt(8 * self.charge_energy * self.joseph_energy)

    @property
    def flux_zpf(self) -> float:
        return (2 * self.charge_energy / self.joseph_energy) ** 0.25

    @property
    def charge_zpf(self) -> float:
        return (self.joseph_energy / (32 * self.charge_energy)) ** 0.25

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
            freq=self.freq,
            anham=self.anharm,
            ext_flux=self.ext_flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        osc_hamil = self.freq * self.basis.num_op

        low_op = self.basis.low_op
        raise_op = self.basis.raise_op

        anharm_op = raise_op @ raise_op @ low_op @ low_op

        hamil = osc_hamil + 0.5 * self.anharm * anharm_op
        return hamil

    def hamiltonian(self, *, as_qobj=False) -> np.ndarray:
        hamil = self.basis.finalize_op(self._get_hamiltonian())

        if as_qobj:
            dim = self.basis.sys_truncated_dims
            qobj_op = Qobj(
                inpt=hamil, dims=[dim, dim], shape=hamil.shape, type="oper", isherm=True
            )
            return qobj_op
        return hamil

    def potential(self, phase: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pot = -self.joseph_energy * np.cos(phase)
        return pot

    def wave_function(self) -> np.ndarray:
        raise NotImplementedError

    def dielectric_loss(self) -> List[np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def from_energies(
        label: str,
        charge_energy: float,
        joseph_energy: float,
        ext_flux: Optional[float] = 0,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> "SimpleTransmon":
        anharm = -charge_energy
        res_freq = np.sqrt(8 * charge_energy * joseph_energy)
        freq = res_freq + anharm
        return SimpleTransmon(
            label,
            freq,
            anharm,
            ext_flux,
            basis=basis,
            dim_hilbert=dim_hilbert,
        )
    def dielectric_rates(
        self, 
        level_k: int, 
        level_m: int, 
        qdiel: float, 
        beta: float, 
        return_op=False
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

        return relaxation_rate, excitation_rate, gamma

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
