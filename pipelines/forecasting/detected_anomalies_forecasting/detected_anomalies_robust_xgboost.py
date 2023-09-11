import os

import numpy as np
import xarray as xr

from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts.modules import ClockShift, CalendarExtraction, CalendarFeature
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper


def forecast(hparams):
    """
    Calculates a forecast using a Linear Regression that considers information on detected anomalies
    """
    pipeline = Pipeline(name='svr', path=os.path.join('run', hparams.name, 'forecasting', 'detected_anomalies_lr'))

    #####
    # Scale load
    ###
    load_scaler = SKLearnWrapper(module=StandardScaler(), name='load_scaler')
    load_scaled = load_scaler(x=pipeline['y_hat'])
    load_scaled = FunctionModule(lambda x: {'load_scaled': xr.DataArray(
            x.values.flatten(), dims=['index'], coords={'index': x['index']}
        )
    }, name='load_scaled')(x=load_scaled)

    #####
    # Shift load by one to exclude actual value from features (i.e. shift value we want to predict out of the data)
    ###
    # NOTE: Shifting will add zeros to the end or the beginning of the data
    factor = int(1 / hparams.resolution)
    load_history = {}
    load_target = {}
    anomalies_history = {}
    for i in range(24 * factor):
        load_target[f'load_target{i}'] = ClockShift(lag=-1 * i, name='load_target_shift')(x=load_scaled)
        load_history[f'load_history{i}'] = ClockShift(lag=(i + 1), name='load_history_shift')(x=load_scaled)
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

    #####
    # Create calendar features
    ###
    calendar = CalendarExtraction(country='Germany',
                                  features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                            CalendarFeature.day_sine, CalendarFeature.monday, CalendarFeature.tuesday,
                                            CalendarFeature.wednesday, CalendarFeature.thursday,
                                            CalendarFeature.friday, CalendarFeature.hour_cos, CalendarFeature.day_cos,
                                            CalendarFeature.month_cos, CalendarFeature.saturday, CalendarFeature.sunday,
                                            CalendarFeature.workday])(x=pipeline['y_hat'])

    #####
    # Define model
    ###
    forecast = SKLearnWrapper(
        module=XGBRegressor(n_jobs=2),
        name='forecast'
    )(
        **load_history,
        **anomalies_history,
        calendar_features=calendar,
        target_load=load_target
    )

    ####
    # Inverse Scale Features
    ###
    # Rescale load values to calculate metrics on original data.
    inverse_scaler = load_scaler(x=forecast, computation_mode=ComputationMode.Transform, use_inverse_transform=True)

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(
        lambda x: {
            'forecast': xr.DataArray(
                x.values,
                dims=['index', 'forecast_horizon'],
                coords={'index': x['index'],
                        'forecast_horizon': range(24 * factor)}
            )
        }, name='forecast')(x=inverse_scaler)

    return pipeline