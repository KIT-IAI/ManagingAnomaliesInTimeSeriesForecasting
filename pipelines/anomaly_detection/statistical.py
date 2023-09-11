import os
import numbers

import pandas as pd
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


global mean, std
mean = None
std = None


def determine_anomalies(x, selected_range=2):
    """
    Determine anomalies based on mean + range * std while the mean can refer to different periods of time

    :param x: Dataset to be considered
    :param selected_range: Factor of the standard deviation that determines which values should be considered as normal
    :return:
    """
    global mean, std
    if isinstance(mean, numbers.Number):
        # overall mean and std approach
        selected_mean = mean
        selected_std = std
    elif len(mean) == 7:
        # weekday based mean and std approach
        filter = x.index.dt.weekday
        selected_mean = mean[filter]
        selected_std = std[filter]
    elif len(mean) == 12:
        # month based mean and std approach
        filter = x.index.dt.month - 1
        selected_mean = mean[filter]
        selected_std = std[filter]
    elif len(mean) == 24:
        # hour based mean and std approach
        filter = x.index.dt.hour
        selected_mean = mean[filter]
        selected_std = std[filter]
    elif len(mean) == 31:
        # day based mean and std approach
        filter = x.index.dt.day - 1
        selected_mean = mean[filter]
        selected_std = std[filter]
    elif len(mean) == 24 * 7:
        # hour based mean and std for each weekday approach
        filter = x.index.dt.weekday * 24 + x.index.dt.hour
        selected_mean = mean[filter]
        selected_std = std[filter]
    elif len(mean) == 24 * 31:
        # hour based mean and std for each month approach
        filter = (x.index.dt.day - 1) * 24 + x.index.dt.hour
        selected_mean = mean[filter]
        selected_std = std[filter]

    # determine anomalies based on threshold
    anomalies = x > selected_mean + selected_range * selected_std
    anomalies |= x < selected_mean - selected_range * selected_std
    anomalies = anomalies.astype(int)

    return xr.DataArray(anomalies, x.coords)


def get_mean_std(frame, target=None):
    """
    Calculate mean and std for a specific time period such as hour, day, weekday, etc.
    """
    global mean, std
    # WARNING: Works only for hourly data
    if target is None or target.lower() == 'overall' or target.lower() == 'o':
        mean, std = frame.mean(), frame.std()
    else:
        temp = pd.DataFrame()
        temp['time_feature'] = frame
        if target.lower() == 'hour' or target.lower() == 'h':
            x = frame.to_xarray()
            temp['time_feature'] = x.index.dt.hour
        elif target.lower() == 'day' or target.lower() == 'd':
            x = frame.to_xarray()
            temp['time_feature'] = x.index.dt.day - 1
        elif target.lower() == 'weekday' or target.lower() == 'wd':
            x = frame.to_xarray()
            temp['time_feature'] = x.index.dt.weekday
        elif target.lower() == 'hourly_weekday' or target.lower() == 'hwd':
            x = frame.to_xarray()
            temp['time_feature'] = x.index.dt.weekday * 24 + x.index.dt.hour
        elif target.lower() == 'month' or target.lower() == 'm':
            x = frame.to_xarray()
            temp['time_feature'] = x.index.dt.month - 1
        elif target.lower() == 'hourly_month' or target.lower() == 'hm':
            x = frame.to_xarray()
            temp['time_feature'] = (x.index.dt.day - 1) * 24 + x.index.dt.hour
        group = temp.groupby('time_feature')
        mean = group['time_feature'].mean().values.flatten()
        std = group['time_feature'].std().values.flatten()


def detect(hparams):
    """
    Detect anomalies using mean and std
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'statistical'))

    #####
    # Apply mean and std
    ###
    _ = FunctionModule(lambda x: determine_anomalies(x, selected_range=hparams.detect_statistical_threshold),
                       lambda x: get_mean_std(x.to_pandas(), hparams.detect_statistical_target),
                       name='anomalies_hat'
    )(x=pipeline['y_hat'])

    return pipeline
