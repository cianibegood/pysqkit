from typing import Optional, Union, List
from copy import copy

import numpy as np
from scipy import linalg as la
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis

_supported_bases = (FockBasis,)

pi = np.pi


class Transmon(Qubit):
    def __init__(
        self,
        label: str,
        charge_energy: float,
        joseph_energy: float,
        flux: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        self._ec = charge_energy
        self._ej = joseph_energy
        self._flux = flux

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
            self.flux,
            basis=copy(self.basis),
        )
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
    def flux(self) -> float:
        return self._flux

    @flux.setter
    def flux(self, flux: float) -> None:
        self._flux = flux

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
            joseph_energy=self.joseph_energy,
            flux=self.flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        charge_op = self._get_charge_op()
        charge_term = 4 * self._ec * charge_op * charge_op

        flux_op = self._get_flux_op()
        cos_mat = la.cosm(flux_op)
        joseph_term = self.joseph_energy * cos_mat

        hamil = charge_term - joseph_term

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
        pot = -self._ej * np.cos(flux)
        return pot

    def wave_function(self) -> np.ndarray:
        raise NotADirectoryError

    def dielectric_loss(self) -> List[np.ndarray]:
        raise NotImplementedError


class SimpleTransmon(Qubit):
    def __init__(
        self,
        label: str,
        freq: float,
        anharm: float,
        ext_flux: Optional[float] = 0,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        self._freq = freq
        self._anharm = anharm
        self._ext_flux = ext_flux

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
            self.flux,
            basis=copy(self.basis),
        )
        return qubit_copy

    @property
    def freq(self) -> float:
        res_freq = self._freq - self._anharm
        shifted_freq = res_freq * np.sqrt(np.abs(np.cos(pi * self._ext_flux)))
        return shifted_freq + self._anharm

    @freq.setter
    def freq(self, freq_val: float) -> None:
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
        return joseph_energy

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
            flux=self.flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            osc_hamil = self.freq * self.basis.num_op

            low_op = self.basis.low_op
            raise_op = self.basis.raise_op

            anharm_term = 0.5 * self.anharm * (raise_op * raise_op * low_op * low_op)

            hamil = osc_hamil + anharm_term
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
        pot = -self.joseph_energy * np.cos(flux)
        return pot

    def wave_function(self) -> np.ndarray:
        raise NotADirectoryError

    def dielectric_loss(self) -> List[np.ndarray]:
        raise NotImplementedError
