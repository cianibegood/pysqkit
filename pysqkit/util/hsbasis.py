#-----------------------------------------------------------------------------
# The module contains functions to define Hilbert-Schmidt basis
# on qudits. 
#-----------------------------------------------------------------------------

import numpy as np
from pysqkit.util.linalg import hilbert_schmidt
from typing import List, Callable, Union

def weyl(
    xi_x: int,
    xi_z: int,
    d: int
) -> np.ndarray:

    """ 
    Returns the qudit Weyl operator with phase-space points xi_x, xi_z
    in a d x d phase space.
    We take the normalized version of the definition in 
    M. Howard et al, Nature 510, 351 (2014), i.e., divided by
    square root of d.
    """

    if xi_x >= d or xi_z >= d:
        raise ValueError("Qudit Pauli labels out of range.")

    if d <= 2:
        raise ValueError("The qudit dimension d must be larger than 2.")

    omega = np.exp(2*np.pi*1j/d)
    z = np.zeros([d, d], dtype=complex)
    x = np.zeros([d, d], dtype=complex)
    for k in range(0, d):
        z[k, k] = omega**k
        x[np.mod(k + 1, d), k] = 1
    z_pow = np.linalg.matrix_power(z, xi_z)
    x_pow = np.linalg.matrix_power(x, xi_x)

    return 1/np.sqrt(d)*omega**(-(d + 1)/2*xi_x*xi_z)*x_pow.dot(z_pow)

def weyl_by_index(
    i: int,
    d: int
) -> np.ndarray:
    
    """
    Returns the Weyl operator associated with index i in a d x d phase-space
    We order the Weyl operators as i -> (xi_x = i // d, xi_z = i % d).
    """
    
    if i >= d**2:
        raise ValueError("Index i out of range: i < d**2")

    xi_x = i // d
    xi_z = i % d
    return weyl(xi_x, xi_z, d)

def iso_basis(
    i: int,
    input_states:List[np.ndarray],
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:

    """ 
    Returns the hilbert-schmidt basis operator associated with index i, 
    using the input_states as defining basis. The input_states
    can be written in an arbitrary basis, and the operator will 
    thus be expressed in this basis.
    """
    
    d = len(input_states) 

    v = hs_basis(i, d)
    v_iso = 0 
    for n in range(0, d):
        for m in range(0, d):
            ket_bra = np.outer(input_states[n], input_states[m].conj())
            v_iso += v[n, m]*ket_bra 
    return v_iso 


