import os

import numpy as np
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


def last_week_forecast(hparams, y, anomalies):
    """
    Calculates last week's value and anomaly information
    """
    factor = int(1 / hparams.resolution)
    y_hat = np.full((len(y), 24 * factor), 0, dtype=np.float)
    median = np.median(y)

    for i in range(24 * factor):
        forecast_idx = np.arange(-168 * factor + i, len(y) - 168 * factor + i)
        anomalies_idx = np.where(anomalies[forecast_idx] > 0)[0]
        forecast_idx[anomalies_idx] = forecast_idx[anomalies_idx] - factor * 168
        valid_forecast = forecast_idx >= 0
        y_hat[valid_forecast, i] = y[forecast_idx[valid_forecast]]
        y_hat[~valid_forecast, i] = median

    return xr.DataArray(y_hat, dims=['index', 'forecast_horizon'],
                        coords={'index': y.index, 'forecast_horizon': range(24 * factor)})


def forecast(hparams):
    """
    Calculates a forecast using last week's value and anomaly information
    """
    pipeline = Pipeline(name='detected_anomalies_last_week', path=os.path.join('run', hparams.name, 'forecasting',
                                                                               'detected_anomalies_last_week'))

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(lambda y, anomalies: last_week_forecast(hparams, y, anomalies),
                           name='forecast')(y=pipeline['y_hat'], anomalies=pipeline['anomalies_hat'])

    return pipeline
