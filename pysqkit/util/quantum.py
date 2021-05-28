from typing import Union
import numpy as np
from qutip import Qobj


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
