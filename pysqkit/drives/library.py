from typing import Optional

import numpy as np

from ..systems import Qubit, Drive


def microwave_drive(
    qubit: Qubit,
    amp: Optional[float] = None,
    freq: Optional[float] = None,
    phase: Optional[float] = None,
    label: Optional[str] = None,
) -> Drive:
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to be a pysqkit.Qubit instance, "
            "instead got {}".format(type(qubit))
        )

    charge_op = qubit.charge_op()
    drive_params = {}

    param_names = ["amp", "freq", "phase"]
    param_vals = [amp, freq, phase]

    for name, val in zip(param_names, param_vals):
        if val is not None:
            if not isinstance(val, float):
                raise ValueError(
                    "{} expected to float, got {} instead".format(name, val)
                )
            drive_params[name] = val

    def microwave_pulse(time, amp, freq, phase):
        return amp * np.cos(2 * np.pi * freq * time + phase)

    drive = Drive(
        pulse=microwave_pulse,
        operator=charge_op,
        label=label,
    )

    drive.set_params(**drive_params)
    return drive

# It can be improved by making one function that gives arbitrary pulse

def gaussian_microwave_drive(
    qubit: Qubit,
    amp: Optional[float] = None,
    freq: Optional[float] = None,
    phase: Optional[float] = None,
    rise_time: Optional[float] = None,
    plateau_time: Optional[float] = None,
    label: Optional[str] = None,
) -> Drive:
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to be a pysqkit.Qubit instance, "
            "instead got {}".format(type(qubit))
        )

    charge_op = qubit.charge_op()
    drive_params = {}

    param_names = ["amp", "freq", "phase"]
    param_vals = [amp, freq, phase]

    for name, val in zip(param_names, param_vals):
        if val is not None:
            if not isinstance(val, float):
                raise ValueError(
                    "{} expected to float, got {} instead".format(name, val)
                )
            drive_params[name] = val

    def gaussian_microwave_pulse(time, amp, freq, phase, rise_time, plateau_time):
        sigma = rise_time/np.sqrt(2*np.pi)
        flat_amplitude = (1 - np.exp(-rise_time**2/(2*sigma**2)))
        if time < rise_time:
            prefactor = 1/flat_amplitude*(np.exp(-(time - \
                rise_time)**2/(2*sigma**2)) - np.exp(-rise_time**2/(2*sigma**2)))
            return prefactor*np.cos(2*np.pi*freq*time)
        elif time < rise_time + plateau_time:
            return np.cos(2*np.pi*freq*time)
        else:
            prefactor = 1/flat_amplitude*(np.exp(-(time - plateau_time - \
                rise_time)**2/(2*sigma**2)) - np.exp(-rise_time**2/(2*sigma**2)))
            return prefactor*np.cos(2*np.pi*freq*time)

    drive = Drive(
        pulse=gaussian_microwave_pulse,
        operator=charge_op,
        label=label,
    )

    drive.set_params(**drive_params)
    return drive
