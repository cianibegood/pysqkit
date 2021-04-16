import numpy as np
from cphase_sim import basis

ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr_='Only square {} matrices can be transformed: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}'
)


def kraus_to_ptm(kraus, pauli_basis=None, subs_dim_hilbert=None):
    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus extend dimension by one
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim = kraus.shape[1]
    if kraus.shape[1:] != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_hilbert
        if dim != basis_dim:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                kraus.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    kraus.shape, basis_dim))
            vectors = np.prod([basis.gell_mann(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            vectors = basis.gell_mann(dim).vectors

    ptm = np.einsum("xab, zbc, ycd, zad -> xy", vectors, kraus,
                    vectors, kraus.conj(), optimize=True).real
    return ptm


def ptm_to_choi(ptm, pauli_basis=None, subs_dim_hilbert=1):

    dim = ptm.shape[0]
    if ptm.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('PTM', ptm.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_pauli
        if dim != basis_dim:
            raise ValueError(
                ERR_MSGS['basis_dim_mismatch'].format(ptm.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)**2
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    ptm.shape, basis_dim))
            vectors = np.prod([basis.gell_mann(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            dim_hilbert = np.sqrt(dim)
            assert dim == dim_hilbert*dim_hilbert
            vectors = basis.gell_mann(dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')
    choi = np.einsum('ij, ijkl -> kl', ptm, tensor, optimize=True).real
    return choi


def choi_to_ptm(choi, pauli_basis=None, subs_dim_hilbert=None):
    if choi.ndim != 2:
        raise ValueError(ERR_MSGS['wrong_dim'].format(choi.ndim))

    dim = choi.shape[0]
    if choi.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Choi', choi.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_pauli
        if dim != basis_dim:
            raise ValueError(
                ERR_MSGS['basis_dim_mismatch'].format(choi.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)**2
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    choi.shape, basis_dim))
            vectors = np.prod([basis.gell_mann(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            dim_hilbert = np.sqrt(dim)
            assert dim == dim_hilbert*dim_hilbert
            vectors = basis.gell_mann(dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')

    product = np.einsum('ij, lmjk -> lmik', choi, tensor, optimize=True)
    ptm = np.einsum('lmii-> lm', product, optimize=True).real
    return ptm


def choi_to_kraus(choi):
    if choi.ndim != 2:
        raise ValueError(ERR_MSGS['wrong_dim'].format(choi.ndim))

    dim_pauli = choi.shape[0]
    if choi.shape != (dim_pauli, dim_pauli):
        raise ValueError(ERR_MSGS['not_sqr'].format('Choi', choi.shape))

    dim_hilbert = int(np.sqrt(dim_pauli))
    assert dim_pauli == dim_hilbert*dim_hilbert

    einvals, einvecs = np.linalg.eig(choi)
    kraus = np.einsum("i, ijk -> ikj", np.sqrt(einvals.astype(complex)),
                      einvecs.T.reshape(dim_pauli, dim_hilbert, dim_hilbert))
    return kraus


def ptm_to_kraus(ptm):
    choi = ptm_to_choi(ptm)
    kraus = choi_to_kraus(choi)
    return kraus


def kraus_to_choi(kraus):
    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus extend dimension by one
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim_hilbert = kraus.shape[1]
    if kraus.shape[1:] != (dim_hilbert, dim_hilbert):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

    dim_pauli = dim_hilbert * dim_hilbert
    choi = np.einsum("ijk, ilm -> kjml", kraus, kraus.conj()
                     ).reshape(dim_pauli, dim_pauli)
    return choi


def convert_ptm_basis(ptm, cur_basis, new_basis):
    dim = ptm.shape[0]
    if ptm.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('PTM', ptm.shape))

    cur_basis_dim = cur_basis.dim_pauli
    if dim != cur_basis_dim:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            ptm.shape, cur_basis_dim))
    cur_vectors = cur_basis.vectors

    new_basis_dim = new_basis.dim_pauli
    if dim != new_basis_dim:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            ptm.shape, new_basis_dim))
    new_vectors = new_basis.vectors

    converted_ptm = np.einsum("xij, yji, yz, zkl, wlk -> xw", new_vectors,
                              cur_vectors, ptm, cur_vectors, new_vectors, optimize=True).real

    return converted_ptm
