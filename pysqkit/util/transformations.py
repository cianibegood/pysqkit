import numpy as np


def kraus_to_ptm(kraus, basis_in, basis_out):
    dim = basis_in.dim_hilbert
    ptm = np.einsum(
        "iab,kbc,jcd,kad->ij", basis_in.ops, kraus, basis_out.ops, kraus.conj()
    )
    return ptm.real / dim
