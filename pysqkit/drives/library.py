from typing import Callable, Optional, Union
import numpy as np

from ..systems import Qubit, Drive

from . import pulses


def microwave_drive(
    qubit: Qubit,
    label: str,
    pulse: Union[np.ndarray, Callable],
    *,
    pulse_shape: Optional[Callable] = None,
    **drive_params,
) -> Drive:
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to be a pysqkit.Qubit instance, "
            "instead got {}".format(type(qubit))
        )

    charge_op = qubit.charge_op()
    # pulse_func = pulse or pulses.cos_modulation

    hilbert_dim = qubit.basis.sys_truncated_dims

    drive = Drive(
        label=label,
        operator=charge_op,
        pulse=pulse,
        pulse_shape=pulse_shape,
        hilbert_dim=hilbert_dim,
    )

    drive.set_params(**drive_params)
    return drive

def bosonic_drive(
    qubit: Qubit,
    label: str,
    drive_type: str,
    pulse: Union[np.ndarray, Callable],
    *,
    pulse_shape: Optional[Callable] = None,
    **drive_params,
) -> Drive:
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to be a pysqkit.Qubit instance, "
            "instead got {}".format(type(qubit))
        )
    
    supported_type = ['raise', 'low']

    if drive_type not in ['raise', 'low']:
        raise ValueError(
            "Unsupported drive_type"
            "drive_type must be in "
            + str(supported_type))
    
    if drive_type=='raise':
        op = qubit.basis.raise_op
    elif drive_type=='low':
        op = qubit.basis.low_op


    hilbert_dim = qubit.basis.sys_truncated_dims

    drive = Drive(
        label=label,
        operator=op,
        pulse=pulse,
        pulse_shape=pulse_shape,
        hilbert_dim=hilbert_dim,
    )

    drive.set_params(**drive_params)
    return drive


