import numpy as np


def cos_modulation(time, amp, freq, phase):
    return amp * np.cos(2 * np.pi * freq * time + phase)
