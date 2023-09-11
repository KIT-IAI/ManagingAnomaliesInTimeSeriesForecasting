import numpy as np


def get_H(power, index, doy_span = 8):
    doy = index // 96
    min_doy = max(0, doy - doy_span)
    max_doy = min(365, doy + doy_span)
    doys = set(list(range(doy, min_doy, -7)) + list(range(doy, max_doy, 7)))

    tod = index % 96

    H = []
    for d in doys:
        for t in range(-4, 5):  # 1 hour before and after index
            i = d * 96 + tod + t
            if i >= 0 and i < power.shape[0] and not np.isnan(power[i]):
                H.append(power[i])

    return H


def get_D(power):
    forward = np.zeros(power.shape[0])

    if power[0] == np.nan:
        forward[0] = 10000000

    for i in range(1, power.shape[0]):
        if np.isnan(power[i]):
            forward[i] = forward[i - 1] + 1

    backward = np.zeros(power.shape[0])

    if np.isnan(power[-1]):
        backward[-1] = 100000000

    for i in range(power.shape[0] - 2, -1, -1):
        if np.isnan(power[i]):
            backward[i] = backward[i + 1] + 1

    return np.minimum(forward, backward)