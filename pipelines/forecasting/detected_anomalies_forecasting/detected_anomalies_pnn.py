import os

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.preprocessing import StandardScaler

from modules.profile_neural_network import ProfileNeuralNetwork

from pywatts.callbacks import LinePlotCallback
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, RollingGroupBy
from pywatts.modules.feature_extraction.calendar_extraction import CalendarExtraction, CalendarFeature
from pywatts.modules.feature_extraction.rolling_mean import RollingMean
from pywatts.modules.feature_extraction.trend_extraction import TrendExtraction
from pywatts.modules.preprocessing.sampler import Sampler
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


def get_diff(x, profile):
    """
    Calculates difference between x and profile
    """
    return numpy_to_xarray(x.values - profile.values, x)


def add_dims(x):
    """
    Adds the time dimension to input x
    """
    return xr.DataArray(x.values.flatten(), dims=['index'], coords={'index': pd.to_datetime(x['index'])})


def forecast(hparams):
    """
    Calculates a forecast using a profile neural network
    """
    pipeline = Pipeline(name='detected_anomalies_pnn', path=os.path.join('run', hparams.name, 'forecasting',
                                                                         'detected_anomalies_pnn'))

    factor = int(1 / hparams.resolution)
    HORIZON = 24 * factor

    #####
    # Prepare features
    ###
    load_scaler = SKLearnWrapper(module=StandardScaler(), name='load_scaler')
    load_scaled = load_scaler(x=pipeline['y_hat'])
    load_scaled = FunctionModule(lambda x: {'load_scaled': xr.DataArray(
            x.values.flatten(), dims=['index'], coords={'index': x['index']}
        )
    }, name='load_scaled')(x=load_scaled)

    factor = int(1 / hparams.resolution)
    HORIZON = 24 * factor
    profile_moving = RollingMean(window_size=28, group_by=RollingGroupBy.WorkdayWeekend)(x=load_scaled)
    difference = FunctionModule(get_diff)(x=load_scaled, profile=profile_moving)
    trend = TrendExtraction(168 * factor, 5)(x=difference)
    sampled_trend = Sampler(HORIZON)(x=trend)
    sampled_profile_moving = Sampler(HORIZON)(x=profile_moving)
    sampled_difference = Sampler(24 * factor)(x=difference)
    calendar = CalendarExtraction(country='Germany',
                                  features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                            CalendarFeature.day_sine, CalendarFeature.monday, CalendarFeature.tuesday,
                                            CalendarFeature.wednesday, CalendarFeature.thursday,
                                            CalendarFeature.friday, CalendarFeature.hour_cos, CalendarFeature.day_cos,
                                            CalendarFeature.month_cos, CalendarFeature.saturday, CalendarFeature.sunday,
                                            CalendarFeature.workday])(x=pipeline['y_hat'])

    load_target = {}
    anomalies_history = {}
    for i in range(24 * factor):
        load_target[f'load_target{i}'] = ClockShift(lag=-1 * i, name='load_target_shift')(x=load_scaled)
        anomalies_history[f'anomalies_history{i}'] = ClockShift(lag=(i + 1), name='anomalies_history_shift')(x=pipeline['anomalies_hat'])
    load_target = FunctionModule(
        lambda **kwargs: {
            'target_load': xr.DataArray(
                data=np.concatenate([kwargs[f'load_target{i}'].values.reshape(-1, 1)
                                     for i in range(24 * factor)],
                                    axis=1),
                dims=['index', 'forecast_horizon'],
                coords={'index': kwargs['load_target0'].index,
                        'forecast_horizon': range(24 * factor)}
            )
        }
    )(**load_target)
    anomalies_history = FunctionModule(
        lambda **kwargs: {
            'anomalies_history': xr.DataArray(
                data=np.concatenate([kwargs[f'anomalies_history{i}'].values.reshape(-1, 1)
                                     for i in range(24 * factor)],
                                    axis=1),
                dims=['index', 'forecast_horizon'],
                coords={'index': kwargs['anomalies_history0'].index,
                        'forecast_horizon': range(24 * factor)}
            )
        }
    )(**anomalies_history)

    #####
    # Define model
    ###
    pnn = ProfileNeuralNetwork(name='pnn_module')
    pnn_output = pnn(historical_input=sampled_difference, calendar=calendar, profile=sampled_profile_moving,
                     trend=sampled_trend, anomalies_history=anomalies_history, target=load_target)

    #####
    # Scale features inversely
    ###
    # Rescale load values to calculate metrics on original data
    inverse_scaler = load_scaler(x=pnn_output, computation_mode=ComputationMode.Transform, use_inverse_transform=True)

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(
        lambda x: {
            'forecast': xr.DataArray(x.values,
                dims=['index', 'forecast_horizon'],
                coords={'index': x['index'],
                        'forecast_horizon': range(24 * factor)}
                )
        }, name='forecast')(x=inverse_scaler)

    return pipeline
