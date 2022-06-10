import numpy as np
from pysqkit import Qubit, QubitSystem
import cmath
from typing import List, Callable
import numpy as np
from pysqkit.util.linalg import get_mat_elem, \
    tensor_prod, hilbert_schmidt_prod
import pysqkit.util.transformations as trf

def zz(system: QubitSystem) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0] \
        - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def mu_yz_c(
    comp_states: List[np.ndarray], 
    op: np.ndarray, 
    eps: float
) -> float:

    """
    Description
    --------------------------------------------------------------------------
    Returns the CR coefficient mu_{YZ} obtained via projection 
    onto the dressed basis
    """

    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 - yz1))/2*eps/2

def mu_yi_c(
    comp_states: List[np.ndarray], 
    op: np.ndarray, 
    eps: float
) -> float:

    """
    Description
    --------------------------------------------------------------------------
    Returns the direct drive coefficient mu_{YI} obtained via projection 
    onto the dressed basis
    """

    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'] )
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 + yz1))/2*eps/2

def mu_yz_sw(
    jc: float,
    eps: float,
    transm_t: Qubit,
    transm_c: Qubit
) -> float:

    """
    Description
    ---------------------------------------------------------------------
    Returns the cross-resonance coefficient in the linear approximation
    using the result of Magesan-Gambetta
    """

    j_eff = jc*transm_c.charge_zpf*transm_t.charge_zpf
    delta_t = transm_t.anharm
    delta_c = transm_c.anharm
    detuning = transm_c.freq - transm_t.freq
    return j_eff*transm_c.charge_zpf*eps/detuning*\
        (delta_c/(delta_c + detuning))/2

def cr_gate_time(
    cr_coeff: float
):
    return 1/(2*np.pi*cr_coeff)*np.pi/4

def single_qubit_corrections(
    sup_op: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    # WARNING: AC I am really puzzled by this...if I use the analog
    # of this for single-qubit gate I get the wrong result and
    # we should actually use sigma_plus...???

    sigma_m1 = tensor_prod([np.array([[0.0, 0.0], [1.0, 0.0]]), 
                           np.array([[1.0, 0.0], [0.0, 0.0]])])
    sigma_m2 = tensor_prod([np.array([[1.0, 0.0], [0.0, 0.0]]), 
                           np.array([[0.0, 0.0], [1.0, 0.0]])])
    sigma_m1_vec = trf.mat_to_vec(sigma_m1, hs_basis)
    sigma_m2_vec = trf.mat_to_vec(sigma_m2, hs_basis)
    evolved_sigma_m1_vec = sup_op.dot(sigma_m1_vec)
    evolved_sigma_m2_vec = sup_op.dot(sigma_m2_vec)
    evolved_sigma_m1 = trf.vec_to_mat(evolved_sigma_m1_vec, hs_basis)
    evolved_sigma_m2 = trf.vec_to_mat(evolved_sigma_m2_vec, hs_basis)
    phi10 = cmath.phase(hilbert_schmidt_prod(sigma_m1, evolved_sigma_m1))
    phi01 = cmath.phase(hilbert_schmidt_prod(sigma_m2, evolved_sigma_m2))
    p_phi10 = np.array([[1, 0], [0, np.exp(-1j*phi10)]])
    p_phi01 = np.array([[1, 0], [0, np.exp(-1j*phi01)]])
    return tensor_prod([p_phi10, p_phi01])


 