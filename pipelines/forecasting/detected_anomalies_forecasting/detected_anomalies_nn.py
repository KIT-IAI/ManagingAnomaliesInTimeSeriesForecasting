import os

import numpy as np
import xarray as xr
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, CalendarExtraction, CalendarFeature
from pywatts.modules.wrappers import FunctionModule, KerasWrapper, SKLearnWrapper


def get_keras_model(hparams):
    """
    Define model of a simple neural network that considers information on detected anomalies
    """
    input_load = tf.keras.layers.Input(shape=(24 * int(1 / hparams.resolution),), name='input_load')
    input_anomalies = tf.keras.layers.Input(shape=(24 * int(1 / hparams.resolution),), name='input_anomalies')
    input_calendar = tf.keras.layers.Input(shape=(14,), name='input_calendar')
    merged = tf.keras.layers.Concatenate(axis=1)([input_load, input_anomalies, input_calendar])
    hidden_1 = tf.keras.layers.Dense(256, activation='relu', name='hidden_1')(merged)
    hidden_2 = tf.keras.layers.Dense(128, activation='relu', name='hidden_2')(hidden_1)
    output = tf.keras.layers.Dense(24 * int(1 / hparams.resolution), activation='linear', name='target_load')(hidden_2)
    model = tf.keras.Model(inputs=[input_load, input_anomalies, input_calendar], outputs=output)
    return model


def forecast(hparams):
    """
    Calculates a forecast using a simple neural network that considers information on detected anomalies
    """
    pipeline = Pipeline(name='detected_anomalies_nn', path=os.path.join('run', hparams.name, 'forecasting',
                                                                        'detected_anomalies_nn'))

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
    load_history = FunctionModule(
        lambda **kwargs: {
            'load_history': xr.DataArray(
                data=np.concatenate([kwargs[f'load_history{i}'].values.reshape(-1, 1)
                                     for i in range(24 * factor)],
                                    axis=1),
                dims=['index', 'forecast_horizon'],
                coords={'index': kwargs['load_history0'].index,
                        'forecast_horizon': range(24 * factor)}
            )
        }
    )(**load_history)
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
    # Create calendar features
    ###
    calendar = CalendarExtraction(country='Germany',
                                  features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                            CalendarFeature.day_sine, CalendarFeature.monday, CalendarFeature.tuesday,
                                            CalendarFeature.wednesday, CalendarFeature.thursday,
                                            CalendarFeature.friday, CalendarFeature.hour_cos, CalendarFeature.day_cos,
                                            CalendarFeature.month_cos, CalendarFeature.saturday, CalendarFeature.sunday,
                                            CalendarFeature.workday])(x=pipeline['y_hat'])
    # sampled_calendar = Sampler(name='FutureCalendar', sample_size=hparams.forecast_sample_size)(x=calendar)

    #####
    # Define model
    ###
    keras_model = get_keras_model(hparams)
    # keras_model.summary()
    # tf.keras.utils.plot_model(keras_model, "keras_model_with_shape_info.png", show_shapes=True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    forecasting_module = KerasWrapper(
        keras_model,
        compile_kwargs=dict(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        ),
        fit_kwargs=dict(
            epochs=100,
            batch_size=128,
            shuffle=True,
            validation_split=0.2,
            callbacks=[callback]
        )
    )
    forecast = forecasting_module(
        input_load=load_history,
        input_anomalies=anomalies_history,
        input_calendar=calendar,
        target_load=load_target
    )

    #####
    # Scale features inversely
    ###
    # Rescale load values to calculate metrics on original data
    inverse_scaler = load_scaler(x=forecast, computation_mode=ComputationMode.Transform, use_inverse_transform=True)

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
