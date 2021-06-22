import numpy as np 
from scipy import constants

def average_photon(energy, beta):
    if beta < 0 or energy < 0:
        raise ValueError("Energy and inverse temeperature must be positive")
    return 1/(np.exp(beta*energy) - 1)

def thermalenergy_to_temperature(
    en_th: float
) -> float:
    """ 
    It gives the temperature in K associated with the thermal energy 
    given as en_th = k_b temp/h in GHz
    """
    if en_th < 0:
        raise ValueError("The thermal energy must be positive")
    
    return constants.h/constants.k*en_th*1e9 

def temperature_to_thermalenergy(
    temp: float
) -> float:
    """ 
    It gives the thermal energy en_th = kb T/h in GHz associated with
    the temperature T in kelvin
    """
    if temp < 0:
        raise ValueError("The thermal energy must be positive")

    return constants.k*temp/constants.h/1e9




