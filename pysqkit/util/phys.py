import numpy as np
from scipy import constants



def average_photon(
    energy: float, 
    thermal_energy: float,
) -> float:
    if thermal_energy < 0 or energy < 0:
        raise ValueError("Energy and thermal energy must be positive")
    if thermal_energy != 0.0:
        return 1 / (np.exp(energy/thermal_energy) - 1)
    else:
        return 0.0


def thermalenergy_to_temperature(thermal_energy: float) -> float:
    """
    It gives the temperature in Kelvin associated with the thermal energy
    given as en_th = k_b temp/h in GHz
    """
    if thermal_energy < 0:
        raise ValueError("The thermal energy must be positive")

    return constants.h / constants.k * thermal_energy * 1e9


def temperature_to_thermalenergy(temp: float) -> float:
    """
    It gives the thermal energy en_th = kb T/h in GHz associated with
    the temperature T in Kelvin
    """
    if temp < 0:
        raise ValueError("The thermal energy must be positive")

    return constants.k * temp / constants.h / 1e9
