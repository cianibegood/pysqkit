#-----------------------------------------------------------------------------
# The module contains functions to define Hilbert-Schmidt basis
# on qudits. 
#-----------------------------------------------------------------------------

import numpy as np
from typing import List, Callable, Union

# Qudit Weyl basis

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

    if d <= 1 or not isinstance(d, int):
        raise ValueError("The qudit dimension d must be an integer" 
                         " larger than 1.")
 
    if d > 2:
        omega = np.exp(2*np.pi*1j/d)
        z = np.zeros([d, d], dtype=complex)
        x = np.zeros([d, d], dtype=complex)

        for k in range(0, d):
            z[k, k] = omega**k
            x[np.mod(k + 1, d), k] = 1

        z_pow = np.linalg.matrix_power(z, xi_z)
        x_pow = np.linalg.matrix_power(x, xi_x)

        return 1/np.sqrt(d)*omega**(-(d + 1)/2*xi_x*xi_z)*x_pow.dot(z_pow)
    
    elif d == 2:
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        x = np.array([[0, 1], [1, 0]])
        z_pow = np.linalg.matrix_power(z, xi_z)
        x_pow = np.linalg.matrix_power(x, xi_x)

        return 1/np.sqrt(d)*1j**(xi_x*xi_z)*x_pow.dot(z_pow)
        

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

# ----------------------------------------------------------------------------

# Qubit Pauli basis

def pauli(xi: Union[np.ndarray, List]) -> np.ndarray:
    """ 
    Description
    --------------------------------------------------------------------------
    Returns the normalized Pauli operator on n qubits associated with the 
    binary vector xi. 
    """

    n = int(len(xi)/2)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    x_xi = np.linalg.matrix_power(x, xi[0])
    z_xi = np.linalg.matrix_power(z, xi[n])
    f = xi[0]*xi[n]
    for k in range(1, n):
        f += xi[k]*xi[n + k]
        x_xi = np.kron(x_xi, np.linalg.matrix_power(x, xi[k]))
        z_xi = np.kron(z_xi, np.linalg.matrix_power(z, xi[n + k]))
    d_xi = (1j)**f*x_xi.dot(z_xi)
    return d_xi/np.sqrt(2**n)

def decimal_to_binary(
    k: int, 
    nbit: int
    ) -> np.ndarray:
    """
    Description
    --------------------------------------------------------------------------
    Returns the integer k as a binary vector with nbit
    """

    y = np.zeros(nbit, dtype=int)
    iterate = True
    x = np.mod(k, 2)
    y[nbit - 1] = int(x)
    if nbit > 1:
        k = (k - x)/2
        l = 1
        while iterate == True:
            l += 1
            x = np.mod(k, 2)
            y[nbit - l] = int(x)
            k = (k - x)/2
            if k <= 0:
                iterate = False
    return y

def binary_to_decimal(k_bin: np.ndarray) -> int:
    """
    Description
    --------------------------------------------------------------------------
    Returns the integer associated with a binary vector
    """

    n = len(k_bin)
    y = k_bin[n-1]
    for l in range(1, n):
        y += 2**l*k_bin[n -l -1]
    return y

def pauli_by_index(
    i: int,
    d: int
) -> np.ndarray:

    """ 
    Description
    --------------------------------------------------------------------------
    Returns the normalized Pauli operator on n = log_2(d) qubits associated 
    with the integer i 
    """
    
    if np.mod(np.log2(d), 1) != 0.0 or d <= 0:
        raise ValueError("Dimension error: d must be a positive power of 2")
    
    n = int(np.log2(d))
    xi = decimal_to_binary(i, 2*n)
    
    return pauli(xi)

# ----------------------------------------------------------------------------


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



