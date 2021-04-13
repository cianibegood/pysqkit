import numpy as np 

def average_photon(energy, beta):
    if beta < 0 or energy < 0:
        raise ValueError("Energy and inverse temeperature must be positive")
    return 1/(np.exp(beta*energy) - 1)