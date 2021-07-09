import numpy as np
from pysqkit.util.linalg import hilbert_schmidt
from typing import List, Callable, Union


def kraus_to_ptm(kraus, basis_in, basis_out):
    dim = basis_in.dim_hilbert
    ptm = np.einsum(
        "iab,kbc,jcd,kad->ij", basis_in.ops, kraus, basis_out.ops, kraus.conj()
    )
    return ptm.real / dim

def weyl(
    xi_x: int,
    xi_z: int,
    d: int
) -> np.ndarray:

    """ It returns the qudit Weyl operator with index xi_x, xi_z.
    We take the normalized version of the definition in 
    M. Howard et al, Nature 510, 351 (2014) """

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
    
    if i >= d**2:
        raise ValueError("Index i out of range: i < d**2")

    xi_x = i // d
    xi_z = i % d
    return weyl(xi_x, xi_z, d)

def kraus_to_super(
    kraus: Union[np.ndarray, List[np.ndarray]],  
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    
    if isinstance(kraus, list):
        d = kraus[0].shape[0]
        for kraus_op in kraus:
            if kraus_op.shape[0] != d:
                raise ValueError("Incompatible Kraus operator dimensions.")
    else:
        d = kraus.shape[0]

    super_op = np.zeros([d**2, d**2], dtype=complex)
    
    for i in range(0, d**2):
        vec_i = hs_basis(i, d)
        kraus_on_vec_i = 0
        if isinstance(kraus, list):
            for kraus_op in kraus:
                kraus_on_vec_i += \
                    kraus_op.dot(vec_i.dot(kraus_op.conj().T))
        else:
            kraus_on_vec_i = kraus.dot(vec_i.dot(kraus.conj().T))

        for k in range(0, d**2):
            vec_k = hs_basis(k, d)
            super_op[i, k] = hilbert_schmidt(vec_k, kraus_on_vec_i)
    
    return super_op

def rho_to_vector(
    rho: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    
    d = rho.shape[0]

    rho_vec = np.zeros(d**2, dtype=complex)

    for i in range(0, d**2):
        vec_i = hs_basis(i, d)
        rho_vec[i] = hilbert_schmidt(vec_i, rho)
    
    return rho_vec

def vector_to_rho(
    rho_vec: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    
    d = int(np.sqrt(rho_vec.shape[0]))

    rho = 0

    for i in range(0, d**2):
        rho += rho_vec[i]*hs_basis(i, d)
    
    return rho


