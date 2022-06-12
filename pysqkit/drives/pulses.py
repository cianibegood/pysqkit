import numpy as np


def cos_modulation(
    time: float, 
    amp: float, 
    freq:float, 
    phase: float
) -> float:
    return amp * np.cos(2 * np.pi * freq * time + phase)

def constant_pulse(amp: float) -> float:
    return amp
