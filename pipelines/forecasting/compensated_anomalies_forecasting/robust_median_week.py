import os

import numpy as np
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


median_week_profile = None


def train_median_week(hparams, y):
    """
    Calculates median values of each weekday to use them as a forecast
    """
    # get first monday 0h
    first_hour_of_weeks = (y.index.dt.weekday == 0) & (y.index.dt.hour == 0) & (y.index.dt.minute == 0)
    start_idx = np.where(first_hour_of_weeks)[0][0]  # get first starting point

    # rotate time series to start at monday 0h
    rotated_time_series = np.roll(y, -1 * start_idx)

    # calculate median profile by calculating the mean for every timestep i
    global median_week_profile
    width = 168 * int(1 / hparams.resolution)
    median_week_profile = np.zeros(width)
    for i in range(width):
        median = np.median(rotated_time_series[i::width])
        median_week_profile[i] = median


def forecast_median_week(hparams, y):
    """
    Calculates median values of each weekday to use them as a forecast
    """
    # get indexes in profile array
    factor = int(1 / hparams.resolution)
    index_in_profile = 24 * factor * y.index.dt.weekday + factor * y.index.dt.hour \
                     + (y.index.dt.minute / 60 / hparams.resolution).astype(int)

    global median_week_profile
    y_hat = np.zeros((len(y), 24 * factor))
    for i in range(24 * factor):
        idx = (index_in_profile + i) % (168 * factor)
        y_hat[:, i] = median_week_profile[idx]

    # calculating forecast by applying calculated indexes
    return xr.DataArray(y_hat, dims=['index', 'forecast_horizon'],
                        coords={'index': y.index, 'forecast_horizon': range(24 * factor)})


def forecast(hparams):
    """
    Calculates a forecast using the median values of each weekday
    """
    pipeline = Pipeline(name='robust_median_week', path=os.path.join('run', hparams.name, 'forecasting',
                                                                     'robust_median_week'))

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(lambda y: forecast_median_week(hparams, y),
                           lambda y: train_median_week(hparams, y),
                           name='forecast')(y=pipeline['y_target'])

    return pipeline
