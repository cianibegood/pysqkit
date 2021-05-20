from typing import Optional
from copy import copy

import numpy as np
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis

_supported_bases = (FockBasis,)

pi = np.pi


class SimpleTransmon(Qubit):
    def __init__(
        self,
        label: str,
        freq: float,
        anharm: float,
        flux: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        self._freq = freq
        self._anharm = anharm
        self._flux = flux

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
        return self._freq

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
    def flux(self) -> float:
        return self._flux

    @flux.setter
    def flux(self, flux: float) -> None:
        self._flux = flux

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
