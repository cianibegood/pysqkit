#-----------------------------------------------------------------------------
# The module contains functions to convert between different representations
# of density matrices and quantum operations.
#-----------------------------------------------------------------------------

import numpy as np
from pysqkit.util.linalg import hilbert_schmidt
from typing import List, Callable, Union

def kraus_to_ptm(kraus, basis_in, basis_out):
    dim = basis_in.dim_hilbert
    ptm = np.einsum(
        "iab,kbc,jcd,kad->ij", basis_in.ops, kraus, basis_out.ops, kraus.conj()
    )
    return ptm.real / dim

def kraus_to_super(
    kraus: Union[np.ndarray, List[np.ndarray]],  
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:

    """
    Returns the superoperator associated with a quantum operation defined by
    a list of kraus operators in a user-defined Hilbert-Schmidt basis 
    via the function hs_basis.
    """
    
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
            super_op[k, i] = hilbert_schmidt(vec_k, kraus_on_vec_i)
    
    return super_op

def mat_to_vec(
    mat: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:

    """ 
    Gives a matrix mat as a vector in a user-defined 
    Hilbert-Schmidt basis via the function hs_basis.
    """
    
    d = mat.shape[0]

    mat_vec = np.zeros(d**2, dtype=complex)

    for i in range(0, d**2):
        vec_i = hs_basis(i, d)
        mat_vec[i] = hilbert_schmidt(vec_i, mat)
    
    return mat_vec

def vec_to_mat(
    mat_vec: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:

    """ 
    Converts a vectorized matrix mat_vec written in a 
    certain Hilbert-Schmidt basis, defined by the function hs_basis, 
    in a standard matrix representation.
    """
    
    d = int(np.sqrt(mat_vec.shape[0]))

    mat = 0

    for i in range(0, d**2):
        mat += mat_vec[i]*hs_basis(i, d)
    
    return mat


