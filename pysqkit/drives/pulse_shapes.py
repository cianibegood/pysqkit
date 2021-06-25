import numpy as np


def pdf(x):
    return np.exp(-(x ** 2) / 2)


def gaussian_edge(time, width):
    std = width / np.sqrt(2 * np.pi)
    const_offset = pdf(width / std)
    pulse = pdf((time - width) / std) - pdf(width / std)
    return pulse / (1 - const_offset)


def gaussian_top(time: np.ndarray, rise_time: float):
    max_time = np.max(time)

    if max_time < 2 * rise_time:
        raise ValueError(
            "Total time {} is less then twice the rise time {}".format(
                max_time, rise_time
            )
        )

    _rise_times = time < rise_time
    _fall_times = time >= max_time - rise_time

    shape = np.ones(len(time))
    shape[_rise_times] = gaussian_edge(time[_rise_times], rise_time)
    shape[_fall_times] = gaussian_edge(
        time[_fall_times] - (max_time - 2 * rise_time), rise_time
    )

    return shape
