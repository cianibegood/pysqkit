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

    drive = Drive(
        operator=charge_op,
        pulse=pulse or pulses.cos_modulation,
        label=label,
        pulse_shape=pulse_shape,
    )

    drive.set_params(**drive_params)
    return drive
