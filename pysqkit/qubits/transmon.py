from typing import Optional, Union, List, Tuple
from copy import copy
from warnings import warn
from itertools import combinations

import numpy as np
from scipy import linalg as la
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis
from ..util.linalg import get_mat_elem
from ..util.phys import average_photon, temperature_to_thermalenergy

_supported_bases = (FockBasis,)

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
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
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

    def flux_op(
        self,
        *,
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
    ) -> np.ndarray:
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

    def hamiltonian(
        self,
        *,
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
    ) -> np.ndarray:
        op = self._get_hamiltonian()
        hamil = self.basis.finalize_op(op, expand=expand)

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
        raise NotImplementedError

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
        diel_loss_tan: Optional[float] = 0.0,
        env_thermal_energy: Optional[float] = 0.0,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        if max_freq < 0:
            raise ValueError("Frequency expected to be non-negative.")
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

        self.diel_loss_tan = diel_loss_tan
        self.env_thermal_energy = env_thermal_energy

        if basis is None:
            basis = fock_basis(dim_hilbert)
        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")
        super().__init__(label=label, basis=basis)

        self._loss_rates = dict(dielectric=self.dielectric_rates)

    def __copy__(self) -> "SimpleTransmon":
        qubit_copy = self.__class__(
            self.label,
            self.freq,
            self.anharm,
            self.ext_flux,
            self.diel_loss_tan,
            self.env_thermal_energy,
            basis=copy(self.basis),
        )
        qubit_copy._drives = {
            label: copy(drive) for label, drive in self._drives.items()
        }
        return qubit_copy

    @property
    def freq(self) -> float:
        res_freq = self._freq - self._anharm
        shifted_freq = res_freq * np.sqrt(np.abs(np.cos(np.pi * self._ext_flux)))
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
        det_joseph_energy = joseph_energy * np.abs(np.cos(np.pi * self.ext_flux))
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

    @property
    def loss_channels(self):
        return list(self._loss_rates.keys())

    def _get_charge_op(self):
        charge_op = 1j * self.charge_zpf * (self.basis.raise_op - self.basis.low_op)

        return charge_op

    def charge_op(
        self,
        *,
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
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

    def flux_op(
        self,
        *,
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
    ) -> np.ndarray:
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

    def hamiltonian(
        self,
        *,
        expand: Optional[bool] = True,
        as_qobj: Optional[bool] = False,
    ) -> np.ndarray:
        op = self._get_hamiltonian()
        hamil = self.basis.finalize_op(op, expand=expand)

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

    def dielectric_rates(
        self,
        level_k: int,
        level_m: int,
    ) -> Tuple[float, float]:
        if not isinstance(self.diel_loss_tan, float):
            raise ValueError(
                "Dielectric loss tangent expected as a"
                "float value, instead got {}".format(type(self.diel_loss_tan))
            )
        if not isinstance(self.env_thermal_energy, float):
            raise ValueError(
                "Environment thermal energy expected as a"
                "float value, instead got {}".format(\
                    type(self.env_thermal_energy))
            )


        if self.diel_loss_tan < 0 or self.env_thermal_energy < 0:
            raise ValueError(
                "Loss tangen and (absolute) "
                "thermal energy kb*T must be positive."
            )

        if level_k == level_m:
            raise ValueError(
                "Eigenstate indices level_k and level_m must be different."
            )

        if level_k < 0 or level_m < 0:
            raise ValueError("Eigenstate indices level_k and level_m must be positive.")
        elif level_k >= self.dim_hilbert or level_m >= self.dim_hilbert:
            raise ValueError(
                "Eigenstate labels k and m must be smaller than"
                " the Hilbert space dimension."
            )
        
        if self.diel_loss_tan == 0:
            return 0.0, 0.0

        if level_k > level_m:
            level_k, level_m = level_m, level_k

        eig_en, eig_vec = self.eig_states([level_k, level_m], expand=False)
        energy_diff = (eig_en[1] - eig_en[0]) / self.charge_energy

        op = self.flux_op(expand=False)
        phi_km = np.abs(get_mat_elem(op, eig_vec[1], eig_vec[0]))

        gamma = self.diel_loss_tan*self._ec * energy_diff ** 2 * phi_km ** 2 / 4
        nth = average_photon(energy_diff * self._ec, self.env_thermal_energy)

        relaxation_rate = gamma * (nth + 1)
        excitation_rate = gamma * nth

        return relaxation_rate, excitation_rate

    def loss_rates(
        self,
        level_k: int,
        level_m: int,
        loss_channels: Optional[List[str]] = None,
    ):
        if loss_channels is not None:
            for channel in loss_channels:
                if channel not in self.loss_channels:
                    raise ValueError(
                        "The provided channel {} is not supported "
                        "by the fluxonium qubit.".format(channel)
                    )

        channels = loss_channels or self.loss_channels

        channel_loss_rates = [
            self._loss_rates[channel](level_k, level_m) for channel in channels
        ]
        total_rates = tuple(map(sum, zip(*channel_loss_rates)))
        return total_rates

    def _get_loss_ops(
        self,
        level_k: int,
        level_m: int,
        loss_channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray]:
        rates = self.loss_rates(level_k, level_m, loss_channels)
        _, states = self.eig_states((level_k, level_m), expand=False)
        trans_ops = _get_trans_ops(*states)
        jump_ops = (np.sqrt(rate) * op for rate, op in zip(rates, trans_ops))
        return jump_ops

    def loss_ops(
        self,
        level_k: int,
        level_m: int,
        loss_channels: Optional[List[str]] = None,
        as_qobj: Optional[bool] = False,
        *,
        expand: Optional[bool] = True,
    ) -> List[np.ndarray]:
        ops = self._get_loss_ops(level_k, level_m, loss_channels)
        if expand:
            ops = (self.basis.expand_op(op) for op in ops)
        if as_qobj:
            qobj_ops = (self._qobj_oper(op, isherm=True) for op in ops)
            return qobj_ops
        return ops

    def collapse_ops(
        self,
        loss_channels: Optional[List[str]] = None,
        as_qobj: Optional[bool] = False,
        *,
        expand: Optional[bool] = True,
    ):
        collapse_ops = []
        level_pairs = combinations(range(self.dim_hilbert), 2)
        for level_pair in level_pairs:
            collapse_ops.extend(
                self.loss_ops(*level_pair, loss_channels, as_qobj, expand=expand)
            )
        return collapse_ops

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


def _get_trans_ops(in_state, out_state):
    down_op = np.outer(in_state, out_state.conj())
    up_op = np.transpose(down_op.conj())
    return down_op, up_op
