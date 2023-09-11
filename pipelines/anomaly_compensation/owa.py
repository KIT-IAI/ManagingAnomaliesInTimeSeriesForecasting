import os
import numpy as np
import xarray as xr

from pywatts.modules import FunctionModule
from pywatts_pipeline.core.pipeline import Pipeline


def get_H(power, index, doy_span=8):
    """
    calculates historical samples
    """
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
    """
    calculates the weights
    """
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


def owa_compensate(x):
    """""
    Calculates the optimal weighted average of Linear Interpolation und Historical Average,
    see Peppanen et al. (2016)
    """

    alpha = 0.1
    power = x

    if power[0] == np.nan:
        power[0] = power.mean()
    linear = power.copy()
    linear = linear.to_dataframe()
    linear.interpolate(inplace=True)
    linear = linear.to_numpy().flatten()

    y = power.copy()
    D = get_D(power)

    doy_span = 8
    for i in range(power.shape[0]):
        if np.isnan(power[i]):
            H = get_H(power, i)
            while len(H) == 0:
                doy_span += 7
                H = get_H(power, i, doy_span)
            doy_span = 8
            ha = sum(H) / len(H)
            w_i = np.exp(-1 * alpha * D[i])
            # print(w_i)
            y[i] = w_i * linear[i] + (1 - w_i) * ha

    return xr.DataArray(
        data=y.values.flatten(), dims=["index"], coords=dict(index=x.index.values)
    )


def compensate(hparams):
    """
    Compensate anomalies with owa (optimal weighted average)
    """

    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'compensation', 'owa'))
    #####
    # Apply compensation model
    ###
    FunctionModule(lambda x: owa_compensate(x), name='y_hat_comp')(x=pipeline['y_hat_comp'])
    return pipeline
