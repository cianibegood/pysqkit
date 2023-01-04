from typing import Union, Optional, List, Tuple, Dict
import warnings
from copy import copy
from itertools import combinations

import numpy as np
from scipy import linalg as la
from scipy import special as ss
import xarray as xr
from qutip import Qobj

from ..systems import Qubit
from ..bases import fock_basis, charge_rotor_basis, FockBasis, ChargeRotorBasis, OperatorBasis
from ..util.linalg import get_mat_elem


_supported_bases = (FockBasis, ChargeRotorBasis)


class Zeropi:
    
    """
    Description
    --------------------------------------------------------------------------
    Class associated with the 0-pi qubit first introduced in
    P. Brooks, A. Kitaev, J. Preskill, Phys. Rev. A 87, 052306 (2013) and 
    further studied in J. Dempster et al, Phys. Rev. B 90, 094518 (2014). 

    We schematically represent the circuit of the 0-pi qubit 
    establishing also our conventions. Each node has a capacitance to 
    ground C_g not shown in the scheme

    phi_1 ------ C_J, E_J ------ phi_2
      |   \         ______________/ |            
      |    |_______|________        |
      C        ____|        |       C  
      |    ___L             L___    |   
      |   /                      \  | 
    phi_3 ------ C_J, E_J ------ phi_4

    We work with new variables 

    phi = 1/2(phi_1 + phi_2 - phi_3 - phi_4)
    chi = 1/2(phi_1 - phi_2 + phi_3 - phi_4)  
    theta = 1/2(phi_1 - phi_2 - phi_3 - phi_4)
    Sigma = 1/2(phi_1 + phi_2 + phi_3 + phi_4)

    In terms of these variables the Hamiltonian becomes 

    H = 4 E_{C Sigma} q_{Sigma}**2 + 4 E_{C chi} q_{chi}**2 + 
        4 E_{C phi} q_{phi}**2 + 4 E_{C theta} q_{theta}**2 + 
        E_L chi**2 + E_L phi**2 - 2 E_J cos(theta) cos(phi)

    with charging energies

    E_{C Sigma} = e**2/(2 C_g),    E_{C chi} = e**2/(2(2C + C_g)),
    E_{C phi} = e**2/(2(2C_J + C_g)), E_{C theta} = e**2/(2(2C + 2 C_J + C_g))

    and inductive energy

    E_L = Phi_0**2/(4 pi**2 L),    Phi_0 = h/(2 e).  
 
    Since only the mode phi and theta are coupled we will only need to 
    treat these modes numerically, while for Sigma and chi we can resort
    to analytical formulas. 
    We will assume the following ordering in the tensor products

    phi \otimes theta.

    """

    def __init__(
        self,
        label: str,
        charge_energy_sigma: float,
        charge_energy_chi: float,
        charge_energy_phi: float,
        charge_energy_theta: float,
        induct_energy: float,
        joseph_energy: float,
        *,
        basis: Optional[Dict[str, OperatorBasis]] = None,
        dim_hilbert: Optional[Dict[str, int]] = {'phi': 50, 'theta': 51},
    ) -> None:
        
        """
        Parameters
        ----------------------------------------------------------------------
        label: str
            A string that identifies the qubit
        charge_energy_sigma: float
            Charging energy of the Sigma mode (uncoupled free particle)
        charge_energy_chi: float
            Charging energy of the chi mode (uncoupled harmonic oscillator)
        charge_energy_phi: float
            Charging energy of the phi mode (harmonic oscillator 
            coupled to the theta mode)
        charge_energy_theta: float
            Charging energy of the theta mode (rotor mode coupled to phi mode)
        induct_energy: float
            Inductive energy
        joseph_energy: float
            Josephson energy
        basis: Optional[Dict[str, OperatorBasis]] = None
            Dictionary of basis in which operators are written. The dictionary
            must have keys 'phi' and 'theta'. If not differenly specified 
            it is assumed by default that phi is treated in the Fock basis
            and theta in the (rotor) charge basis.
        dim_hilbert: Optional[Dict[str, int]] = {'phi': 50, 'theta': 51}
            Dictionary that stores the Hilbert space dimension for 
            modes phi and theta.       
        """

        self._ec_sigma = charge_energy_sigma
        self._ec_chi = charge_energy_chi
        self._ec_phi = charge_energy_phi
        self._ec_theta = charge_energy_theta
        self._el = induct_energy
        self._ej = joseph_energy

        if basis is None:
            basis = {}
            basis["phi"] = fock_basis(dim_hilbert['phi'])
            basis["theta"] = charge_rotor_basis(dim_hilbert["theta"])
        else:
            if not isinstance(basis, _supported_bases):
                raise NotImplementedError("Basis not supported yet")
        

    def __copy__(self) -> "Zeropi":
        qubit_copy = self.__class__(
            self.label,
            self.charge_energy_sigma,
            self.charge_energy_chi,
            self.charge_energy_phi,
            self.charge_energy_theta,
            self.induct_energy,
            self.joseph_energy,
            basis=copy(self.basis),
        )
        qubit_copy._drives = {
            label: copy(drive) for label, drive in self._drives.items()
        }
        return qubit_copy

    @property
    def charge_energy_sigma(self) -> float:
        return self._ec_sigma

    @charge_energy_sigma.setter
    def charge_energy_sigma(self, charge_energy_sigma: float) -> None:
        self._ec_sigma = charge_energy_sigma
    
    @property
    def charge_energy_chi(self) -> float:
        return self._ec_chi

    @charge_energy_chi.setter
    def charge_energy_chi(self, charge_energy_chi: float) -> None:
        self._ec_chi = charge_energy_chi
    
    @property
    def charge_energy_phi(self) -> float:
        return self._ec_phi

    @charge_energy_phi.setter
    def charge_energy_phi(self, charge_energy_phi: float) -> None:
        self._ec_phi = charge_energy_phi
    
    @property
    def charge_energy_theta(self) -> float:
        return self._ec_theta

    @charge_energy_theta.setter
    def charge_energy_theta(self, charge_energy_theta: float) -> None:
        self._ec_theta = charge_energy_theta
    
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
    def mass_sigma(self) -> float:
        return 1/(8*self._ec_sigma)
    
    @property
    def mass_chi(self) -> float:
        return 1/(8*self._ec_chi)
    
    @property
    def mass_phi(self) -> float:
        return 1/(8*self._ec_phi)
    
    @property
    def mass_theta(self) -> float:
        return 1/(8*self._ec_theta)
    
    @property
    def res_freq_chi(self) -> float:
        return (8*self._ec_chi/(2*self._el))**0.25
    
    @property
    def res_freq_phi(self) -> float:
        return (8*self._ec_phi/(2*self._el))**0.25
    
    @property
    def flux_zpf_chi(self) -> float:
        return (2*self._ec_chi/(2*self._el))**0.25
    
    @property
    def flux_zpf_phi(self) -> float:
        return (2*self._ec_phi/(2*self._el))**0.25
    
    @property
    def _qubit_attrs(self) -> dict:
        q_attrs = dict(
            charge_energy_sigma=self.charge_energy_sigma,
            charge_energy_chi=self.charge_energy_chi,
            charge_energy_phi=self.charge_energy_phi,
            charge_energy_theta=self.charge_energy_theta,
            joseph_energy=self.joseph_energy,
            induct_energy=self.induct_energy
        )
        return q_attrs
    
    def collapse_ops(self):
        return None 
    
    def dephasing_op(self):
        return None

    def _get_Hamiltonian(self) -> np.ndarray:
        pass 
    
    def hamiltonian(self, *, expand_op=True) -> np.ndarray:
        return None
    
    def potential(self) -> np.ndarray:
        return None
    
    def wave_function(self) -> np.ndarray:
        return None
    


    


    
        

def _get_trans_ops(in_state, out_state):
    down_op = np.outer(in_state, out_state.conj())
    up_op = np.transpose(down_op.conj())
    return down_op, up_op
