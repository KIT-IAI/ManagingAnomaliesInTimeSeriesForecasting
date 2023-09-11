import os

import numpy as np
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


def last_day_forecast(hparams, y):
    """
    Shift time series by one to the right and use that as a forecast
    """
    factor = int(1 / hparams.resolution)
    y_hat = np.zeros((len(y), 24 * factor), dtype=float)
    median = np.median(y)

    for i in range(24 * factor):
        forecast_idx = np.arange(-24 * factor + i, len(y) - 24 * factor + i)
        valid_forecast = forecast_idx >= 0
        y_hat[valid_forecast, i] = y[forecast_idx[valid_forecast]]
        y_hat[~valid_forecast, i] = median

    return xr.DataArray(y_hat, dims=['index', 'forecast_horizon'],
                        coords={'index': y.index, 'forecast_horizon': range(24 * factor)})


def forecast(hparams):
    """
    Calculates a forecast using last value
    """
    pipeline = Pipeline(name='last_value', path=os.path.join('run', hparams.name, 'forecasting', 'last_value'))

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(lambda y: last_day_forecast(hparams, y), name='forecast')(y=pipeline['y_target'])

    return pipeline
