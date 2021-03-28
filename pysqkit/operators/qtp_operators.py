import qutip as qtp


def low_op(dim: int) -> qtp.Qobj:
    return qtp.operators.destroy(dim)


def raise_op(dim: int) -> qtp.Qobj:
    return qtp.operators.create(dim)


def num_op(dim: int) -> qtp.Qobj:
    return qtp.operators.num(dim)


def id_op(dim: int) -> qtp.Qobj:
    return qtp.operators.identity(dim)


sigma = {
    'I': qtp.operators.identity(2),
    'X': qtp.operators.sigmax(),
    'Y': qtp.operators.sigmay(),
    'Z': qtp.operators.sigmaz(),
}
