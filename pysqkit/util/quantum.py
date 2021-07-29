from typing import Union, Dict, List
import numpy as np
from qutip import Qobj
from ..systems import QubitSystem
from .linalg import get_mat_elem



def to_oper(state_vec: Union[np.ndarray, Qobj]) -> Union[np.ndarray, Qobj]:
    if isinstance(state_vec, Qobj):
        if state_vec.type != "ket":
            raise ValueError(
                "state_vec type expected to be 'ket', "
                "instead got {}".format(state_vec.type)
            )
        return state_vec * state_vec.dag()
    if isinstance(state_vec, np.ndarray):
        oper = np.einsum("i, j-> ij", state_vec, state_vec.conj())
        return oper
    raise ValueError(
        "state_vec expected to be np.ndarray or qutip.Qobj "
        "instance, instead got {}".format(type(state_vec))
    )

def generalized_rabi_frequency(
    levels: List['str'],
    eps: Dict,
    drive_frequency: float,
    system: QubitSystem
) -> float:
    if len(levels) != 2:
        raise ValueError('The generalized rabi frequency is defined betwen two levels. '
                        'Please specify the desired two levels.')
        
    qubit_labels = system.labels
    drive_op = 0
    for label in qubit_labels:
        drive_op += eps[label]*system[label].charge_op()
    
    in_energy, in_state = system.state(levels[0])
    out_energy, out_state = system.state(levels[1])   
    big_omega_transition = np.abs(get_mat_elem(drive_op, in_state, out_state))
    
    energy_transition = np.abs(in_energy - out_energy)
    drive_detuning = energy_transition - drive_frequency
    
    return np.sqrt(big_omega_transition**2 + drive_detuning**2)

