import numpy as np


def avg_process_fid(ptm, target_ptm):
    dim_pauli = ptm.shape[0]
    dim = np.sqrt(dim_pauli)
    fid = (np.trace(target_ptm.conj().T @ ptm) + dim) / (dim * (dim + 1))
    return fid
