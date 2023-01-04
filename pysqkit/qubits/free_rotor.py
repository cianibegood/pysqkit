from typing import Union, Optional, List
import numpy as np
from copy import copy
from scipy import special as ss
import xarray as xr
from qutip import Qobj

from ..systems import Qubit
from ..bases import charge_rotor_basis, ChargeRotorBasis, OperatorBasis


_supported_bases = (ChargeRotorBasis,)


class FreeRotor(Qubit):
    
    """
    Description
    --------------------------------------------------------------------------
    Class associated with a free rotor

    H = 4 E_C n^2 = 4 E_C \sum_{n=-\infty}^{+\infty} n^2 |n><n|

    with E_C the charging energy
    """

    def __init__(
        self,
        label: str,
        charge_energy: float,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        
        """
        Parameters
        ----------------------------------------------------------------------
        label: str
            A string that identifies the free rotor
        charge_energy: float
            Charging energy of the free rotor
        basis: Optional[OperatorBasis] = None
            Basis in which we want to write the operators. If not provided
            it is assumed it is the Charge Rotor basis
        dim_hilbert: Optional[int] = 51
            Hilbert space dimension       
        """

        self._ec = charge_energy

        if basis is None:
            basis = charge_rotor_basis(dim_hilbert)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)

    def __copy__(self) -> "FreeRotor":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy,
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
    def rotor_mass(self) -> float:
        return 1/(8*self._ec)

    def _get_charge_op(self):
        return self.basis.charge_op

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

    @property
    def _qubit_attrs(self) -> dict:
        q_attrs = dict(
            charge_energy=self.charge_energy,
        )
        return q_attrs

    def _get_hamiltonian(
        self
    ) -> np.ndarray:
        if isinstance(self.basis, ChargeRotorBasis):
            n_op = self.basis.charge_op
            free_rotor_hamil = \
                4*self.charge_energy*n_op.dot(n_op)
        else:
            raise NotImplementedError

        return free_rotor_hamil.real

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
        return None

    def wave_function(
        self,
        phase: Union[float, np.ndarray],
        levels: Optional[Union[int, np.ndarray]] = 0,
        *,
        get_data=False,
    ) -> xr.Dataset:
        if isinstance(levels, int):
            levels = np.array([levels])

        if isinstance(phase, float):
            phase = np.array([phase])
        
        eig_vals = [4*self.charge_energy*x**2 for x in list(levels)]
        eig_vals = np.array(eig_vals)

        prefactor = 1/np.sqrt(2*np.pi)

        wave_func_args = np.einsum('i, k', levels, phase)

        wave_funcs = prefactor*np.exp(1j*wave_func_args)

        if get_data:
            return wave_funcs

        dataset = xr.Dataset(
            data_vars=dict(
                wave_func=(["level", "phase"], wave_funcs),
                energy=(["level"], eig_vals)
            ),
            coords=dict(
                level=levels,
                phase=phase,
            ),
            attrs=dict(
                dim_hilbert=self.dim_hilbert,
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