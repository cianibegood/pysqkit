import numpy as np


def gate_fidelity(ptm, ptm_ideal):
    pauli_dim = ptm.shape[0]
    if ptm.shape != (pauli_dim, pauli_dim):
        raise ValueError(
            "Only square Pauli transfer matrices accepted: provided process matrix is {} dimensional".format(ptm.shape))
    pauli_dim_ideal = ptm_ideal.shape[0]
    if ptm_ideal.shape != (pauli_dim_ideal, pauli_dim_ideal):
        raise ValueError(
            "Only square Pauli transfer matrices accepted: provided ideal matrix is {} dimensional".format(ptm_ideal.shape))
    if pauli_dim != pauli_dim_ideal:
        raise ValueError(
            "The process and ideal transfer matrices must have the same dimensionality: instead process and ideal matrix have dimensions {} and {}".format(pauli_dim, pauli_dim_ideal))

    gate_fid = np.einsum("ji, ji", ptm, ptm_ideal)/pauli_dim
    return gate_fid
