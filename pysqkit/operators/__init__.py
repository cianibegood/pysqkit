_use_qutip = False

if _use_qutip:
    from .qtp_operators import low_op, raise_op, id_op, num_op, sigma
else:
    from .np_operators import low_op, raise_op, id_op, num_op, sigma

__all__ = ['low_op', 'raise_op', 'id_op', 'num_op', 'sigma', '_use_qutip']
