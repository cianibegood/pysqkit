import numpy as np


def pdf(x):
    return np.exp(-(x ** 2) / 2)


def gaussian_edge(time, width):
    std = width / np.sqrt(2 * np.pi)
    const_offset = pdf(width / std)
    pulse = pdf((time - width) / std) - pdf(width / std)
    return pulse / (1 - const_offset)


def gaussian_top(time: np.ndarray, rise_time: float, pulse_time: float, start_time: float=0.0):
    #max_time = np.max(time)

    if pulse_time < 2 * rise_time:
        raise ValueError(
            "Total time {} is less then twice the rise time {}".format(
                pulse_time, rise_time
            )
        )
    
    _start_times = time < start_time
    _rise_times = (time < start_time + rise_time)*(time >= start_time)
    _drive_times = (time >= start_time + rise_time)*(time < start_time + pulse_time - rise_time)
    _fall_times = (time >= start_time + pulse_time - rise_time)*(time < pulse_time + start_time)
    _end_times = time >= pulse_time + start_time

    if rise_time == 0:
        shape = np.ones(len(time))
        shape[_start_times] = 0.0
        shape[_end_times] = 0.0
        return shape

    shape = np.ones(len(time))
    shape[_start_times] = 0.0
    shape[_end_times] = 0.0
    shape[_rise_times] = gaussian_edge(time[_rise_times], rise_time)
    shape[_fall_times] = gaussian_edge(
        time[_fall_times] - (pulse_time - 2*rise_time), rise_time
    )

    return shape
