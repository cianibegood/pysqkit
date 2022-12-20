from typing import Union, Optional, List, Tuple
import warnings
from copy import copy
from itertools import combinations

import numpy as np
from scipy import linalg as la
from scipy import special as ss
import xarray as xr
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, FockBasis, OperatorBasis
from ..util.linalg import get_mat_elem
from ..util.phys import average_photon, temperature_to_thermalenergy


_supported_bases = (FockBasis,)


class Fluxonium(Qubit):
    
    """
    Description
    --------------------------------------------------------------------------
    Class associated with a fluxonium qubit first introduced in
    V. Manucharyan et al., 	Science 326, 113-116 (2009). The fluxonium has
    basic Hamiltonian

    H = 4 E_C q^2 + E_L \phi^2/2 - E_J \cos(\phi + 2 \pi \Phi_ext/\Phi_0)

    with E_C the charging energy, E_L the inductive energy, E_J the 
    Josephson energy, \Phi_ext the external flux, \Phi_0 = h/(2 e) the
    superconducting flux quantum. q is the dimensionless charge 
    and \phi the dimensionless flux and in the quantum setting they 
    satisfy commutation relation [\phi, q] = i
    """

    def __init__(
        self,
        label: str,
        charge_energy: float,
        induct_energy: float,
        joseph_energy: float,
        ext_flux: Optional[float] = 0.5,
        diel_loss_tan: Optional[float] = 0.0,
        freq_loss_tan: Optional[float] = 6.0,
        ind_loss_tan: Optional[float] = 0.0,
        env_thermal_energy: Optional[float] = 0.0, #kb T
        dephasing_times: Optional[dict] = None,
        *,
        basis: Optional[OperatorBasis] = None,
        dim_hilbert: Optional[int] = 100,
    ) -> None:
        
        """
        Parameters
        ----------------------------------------------------------------------
        label: str
            A string that identifies the qubit
        charge_energy: float
            Charging energy of the fluxonium circuit
        induct_energy: float
            Inductive energy of the fluxonium circuit
        joseph_energy: float
            Josephson energy of the fluxonium circuit
        ext_flux: Optional[float] = 0.5,
            External flux in units of \Phi_0. 
        diel_loss_tan: Optional[float] = 0.0
            Dielectric loss tangent that governs relaxation via dielectric
            loss. The value is usually given at 6.0 GHz. The loss tangent 
            follows a law with frequency ~ omega^{epsilon} with epsilon taken 
            to be 0.15. 
            See for instance 
            L. B. Nguyen et al., Phys. Rev. X 9, 041041 (2019).
        freq_loss_tan: Optional[float] = 6.0
            Frequency at which the dielectric loss tangent is given.
        ind_loss_tan: Optional[float] = 0.0
            Inductive loss tangent, i.e., the inverse of the quality factor
            that governs relaxation via inductive losses. See 
            Hazard et al., Phys. Rev. Lett. 122, 010504 (2019) or 
            Nguyen et al., Phys. Rev. X, 9, 041041 (2019).
        env_thermal_energy: Optional[float] = 0.0
            Thermal energy of the environment k_b T.
        dephasing_times: Optional[dict] = None
            A dictionary that stores the pure dephasing times between the 
            specified level and the reference level 0. For reference see
            PRX Quantum 2, 030306 (2021).
            Example:
            dephasing_times = {'1': 100, '3': 10}
            means that the 0-1 transition has pure dephasing time 100
            and the 0-3 transition 10. All the other transitions 
            will be assumed to have 0 pure dephasing time. 
            The unit of measure has to be consistent with the previous data 
            provided by the user.
        basis: Optional[OperatorBasis] = None
            Basis in which we want to write the operators. If not provided
            it is assumed it is the Fock basis
        dim_hilbert: Optional[int] = 100
            Hilber space dimension       
        """

        self._ec = charge_energy
        self._el = induct_energy
        self._ej = joseph_energy
        self._ext_flux = ext_flux

        self.diel_loss_tan = diel_loss_tan
        self.freq_loss_tan = freq_loss_tan
        self.ind_loss_tan = ind_loss_tan
        self.env_thermal_energy = env_thermal_energy
        self.dephasing_times = dephasing_times

        if basis is None:
            basis = fock_basis(dim_hilbert)

        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")

        super().__init__(label=label, basis=basis)
        
        self._loss_rates = dict(dielectric=self.dielectric_rates, inductive=self.inductive_rates)
        self._dephasing_rates = dict(pure_dephasing=self.pure_dephasing_rate)

    def __copy__(self) -> "Fluxonium":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy,
            self.induct_energy,
            self.joseph_energy,
            self.ext_flux,
            self.diel_loss_tan,
            self.freq_loss_tan,
            self.ind_loss_tan,
            self.env_thermal_energy,
            self.dephasing_times,
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
    def ext_flux(self) -> float:
        return self._ext_flux

    @ext_flux.setter
    def ext_flux(self, flux_val: float) -> None:
        self._ext_flux = flux_val

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

    @property
    def loss_channels(self):
        return list(self._loss_rates.keys())
    
    @property
    def dephasing_channels(self):
        return list(self._dephasing_rates.keys())

    def _get_charge_op(self):
        charge_op = 1j * self.charge_zpf * (self.basis.raise_op - self.basis.low_op)
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
            joseph_energy=self.joseph_energy,
            ext_flux=self.ext_flux,
        )
        return q_attrs

    def _get_hamiltonian(
        self,
    ) -> np.ndarray:
        if isinstance(self.basis, FockBasis):
            osc_hamil = self.res_freq*(self.basis.num_op + \
                 0.5*self.basis.id_op)

            flux_phase = np.exp(1j*2*np.pi*self.ext_flux)

            exp_mat = flux_phase * la.expm(1j*self._get_flux_op())
            cos_mat = 0.5 * (exp_mat + exp_mat.conj().T)

            hamil = osc_hamil - self.joseph_energy*cos_mat
        else:
            raise NotImplementedError

        return hamil.real

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
        pot = 0.5 * self._el * flux * flux - self._ej * np.cos(
            flux + 2 * np.pi * self.ext_flux
        )
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

    def dielectric_rates(
        self,
        level_k: int, # Suggestion: shall we pass this to string?
        level_m: int
    ) -> Tuple[float, float]:
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the relaxation and excitation rate due 
        to dielectric loss between two eigenstates
        """
               
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
                "Loss tangent and (absolute) "
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
        energy_diff = (eig_en[1] - eig_en[0])/self._ec

        diel_loss_tan_eff = \
            self.diel_loss_tan*(np.abs(energy_diff)/(self.freq_loss_tan/self._ec))**0.15


        op = self.flux_op(expand=False)
        phi_km = np.abs(get_mat_elem(op, eig_vec[1], eig_vec[0]))

        gamma = diel_loss_tan_eff*self._ec*energy_diff**2*phi_km**2/4
        if self.env_thermal_energy > 0:
            nth = average_photon(energy_diff*self._ec, self.env_thermal_energy)

            relaxation_rate = gamma * (nth + 1)
            excitation_rate = gamma * nth

            return relaxation_rate, excitation_rate
        else:
            return gamma, 0.0
    
    def inductive_rates(
        self,
        level_k: int,
        level_m: int
    ) -> Tuple[float, float]:

        """
        Description
        ----------------------------------------------------------------------
        Returns the relaxation and excitation rate due 
        to inductive loss between two eigenstates.
        """

        if not isinstance(self.ind_loss_tan, float):
            raise ValueError(
                "Inductive loss tangent expected as a"
                "float value, instead got {}".format(type(self.diel_loss_tan))
            )
        if not isinstance(self.env_thermal_energy, float):
            raise ValueError(
                "Environment thermal energy expected as a"
                "float value, instead got {}".format(\
                    type(self.env_thermal_energy))
            )


        if self.ind_loss_tan < 0 or self.env_thermal_energy < 0:
            raise ValueError(
                "Loss tangent and (absolute) "
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
        
        if self.ind_loss_tan == 0:
            return 0.0, 0.0

        if level_k > level_m:
            level_k, level_m = level_m, level_k

        eig_en, eig_vec = self.eig_states([level_k, level_m], expand=False)
        energy_diff = (eig_en[1] - eig_en[0])/self._ec


        op = self.flux_op(expand=False)
        phi_km = np.abs(get_mat_elem(op, eig_vec[1], eig_vec[0]))

        gamma = 2*self.ind_loss_tan*self._el*phi_km**2/4
        if self.env_thermal_energy > 0:
            nth = average_photon(energy_diff*self._ec, self.env_thermal_energy)

            relaxation_rate = gamma * (nth + 1)
            excitation_rate = gamma * nth

            return relaxation_rate, excitation_rate
        else:
            return gamma, 0.0

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
        loss_channels: Optional[List[str]] = None
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
                self.loss_ops(*level_pair, loss_channels, as_qobj, 
                              expand=expand)
            )
        return collapse_ops
    
    def pure_dephasing_rate(
        self, 
        level: int,
    ) -> float:
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the pure dephasing rate between the level and the reference
        ground state

        Warning
        ----------------------------------------------------------------------
        This simply returns the corresponding 1/dephasing_time. 
        Since this is not calculated within the class, the user has to 
        guarantee that it is consistent with the other units of measure. 
        """ 
        
        levels_id = str(level)

        if self.dephasing_times is None:
            return 0.0
        else:
            if levels_id in self.dephasing_times.keys():
                return 1/self.dephasing_times[levels_id]
            else:
                return 0.0 
    
    def dephasing_rate(
        self,
        level,
        dephasing_channels: Optional[List[str]] = None,
    ) -> float:
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the total pure dephasing rate of the level with respect to the 
        reference ground state
        """

        if dephasing_channels is not None:
            for channel in dephasing_channels:
                if channel not in self.dephasing_channels:
                    raise ValueError(
                        "The provided channel {} is not supported "
                        "by the fluxonium qubit.".format(channel)
                    )

        channels = dephasing_channels or self.dephasing_channels

        channel_dephasing_rates = [
            self._dephasing_rates[channel](level) for channel in channels
        ]
        total_rate = sum(channel_dephasing_rates)
        return total_rate
    
    def _get_dephasing_op(
        self,
        dephasing_channels: Optional[List[str]] = None
    ) -> np.ndarray:

        """
        Description
        ----------------------------------------------------------------------
        Returns the multilevel dephasing operator

        Z_dephasing = \sum_{i=1}^{n_levels} \sqrt{2 \gamma_i} |i><i|

        where |i> denotes an eigenstate.
        """
        if self.dephasing_times is None:
            return None
        else:
            rate = {}

            for level in range(1, self.dim_hilbert):
                if self.dephasing_rate(level, dephasing_channels) != 0.0:
                    rate[str(level)] = \
                        self.dephasing_rate(level, dephasing_channels)
            
            deph_op = 0.0
            
            _, eig_vecs = self.eig_states(expand=False) 

            for level_id in rate.keys():
                projector = np.outer(eig_vecs[int(level_id)], 
                                    eig_vecs[int(level_id)].conj())
                
                deph_op += np.sqrt(2*rate[level_id])*projector
            
            return deph_op    
    
    def dephasing_op(
        self,
        dephasing_channels: Optional[List[str]] = None,
        as_qobj: Optional[bool] = False,
        *,
        expand: Optional[bool] = True
    ) -> Union[np.ndarray, Qobj]:

        """
        Description
        ----------------------------------------------------------------------
        Returns the multilevel dephasing operator from the method 

        _get_dephasing_op

        The operator is either returned as a numpy array or as a qutip Qobj.        
        """
        
        op = self._get_dephasing_op(dephasing_channels)

        if op is None:
            return None
        else:
            if expand:
                op = self.basis.expand_op(op)
            
            if as_qobj:
                qobj_op = self._qobj_oper(op, isherm=True)
                return qobj_op
            return op
        

def _get_trans_ops(in_state, out_state):
    down_op = np.outer(in_state, out_state.conj())
    up_op = np.transpose(down_op.conj())
    return down_op, up_op
