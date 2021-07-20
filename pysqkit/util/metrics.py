import numpy as np


def average_process_fidelity(
    sup_op1: np.ndarray,
    sup_op2: np.ndarray
) -> float:
    
    """
    Returns the average process fidelity between two superoperators 
    sup_op1, sup_op2 as defined in Wood-Gambetta, Phys. Rev. A 97, 
    032306 (2018) Eq. 8.
    """
    d = np.sqrt(sup_op1.shape[0])
    return np.real(np.trace(sup_op1.conj().T.dot(sup_op2))/d**2)

def average_gate_fidelity(
    sup_op1: np.ndarray,
    sup_op2: np.ndarray,
    l1: float
) -> float:

    """
    Returns the average gate fidelity between two superoperators 
    sup_op1, sup_op2 with average leakage l1 as 
    defined in Wood-Gambetta, Phys. Rev. A 97, 032306 (2018) Eq. 6.
    """

    d = np.sqrt(sup_op1.shape[0]) 
    f_pro = average_process_fidelity(sup_op1, sup_op2)
    return (d*f_pro + 1 - l1)/(d + 1)
    
    
    
