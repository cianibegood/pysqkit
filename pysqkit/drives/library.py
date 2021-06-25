from typing import Callable, Optional

from ..systems import Qubit, Drive

from . import pulses


def microwave_drive(
    qubit: Qubit,
    label: str,
    *,
    pulse: Optional[Callable] = None,
    pulse_shape: Optional[Callable] = None,
    **drive_params,
) -> Drive:
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to be a pysqkit.Qubit instance, "
            "instead got {}".format(type(qubit))
        )

    charge_op = qubit.charge_op()
    pulse_func = pulse or pulses.cos_modulation

    hilbert_dim = qubit.basis.sys_truncated_dims

    drive = Drive(
        label=label,
        operator=charge_op,
        pulse=pulse_func,
        pulse_shape=pulse_shape,
        hilbert_dim=hilbert_dim,
    )

    drive.set_params(**drive_params)
    return drive
