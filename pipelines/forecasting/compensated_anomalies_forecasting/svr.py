import os

import numpy as np
import xarray as xr

from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts.modules import ClockShift, CalendarExtraction, CalendarFeature
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper


def forecast(hparams):
    """
    Calculates a forecast using a SVR
    """
    pipeline = Pipeline(name='svr', path=os.path.join('run', hparams.name, 'forecasting', 'svr'))

    #####
    # Scale load
    ###
    load_scaler = SKLearnWrapper(module=StandardScaler(), name='load_scaler')
    load_scaled = load_scaler(x=pipeline['y_target'])
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
    for i in range(24 * factor):
        load_target[f'load_target{i}'] = ClockShift(lag=-1 * i, name='load_target_shift')(x=load_scaled)
        load_history[f'load_history{i}'] = ClockShift(lag=(i + 1), name='load_history_shift')(x=load_scaled)
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
                                            CalendarFeature.workday])(x=pipeline['y_target'])

    #####
    # Define model
    ###
    par = {'C': 1.0, 'epsilon': 0.1, 'tol': 1e-4, 'max_iter': 10000}
    new_par = {}
    for key in par.keys():
        h_key = "svr#" + key
        new_par[key] = getattr(hparams, h_key) if (h_key in hparams) else par[key]
    forecast = SKLearnWrapper(
        module=MultiOutputRegressor(LinearSVR(**new_par), n_jobs=2),        # unpack dictionary to transfer keyword arguments
        name='forecast'
    )(
        **load_history,
        calendar_features=calendar,
        target=load_target
    )

    ####
    # Inverse Scale Features
    ###
    # Rescale load values to calculate metrics on original data.
    inverse_scaler = load_scaler(x=forecast, computation_mode=ComputationMode.Transform, use_inverse_transform=True)

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(lambda x: {'forecast': xr.DataArray(
            x.values,
            dims=['index', 'forecast_horizon'],
            coords={'index': x['index'],
                    'forecast_horizon': range(24 * factor)}
        )
    }, name='forecast')(x=inverse_scaler)

    return pipeline
