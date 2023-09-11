import os

import xarray as xr

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, CalendarExtraction, CalendarFeature
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper, SmTimeSeriesModelWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


def forecast(hparams):
    """
    Calculates a forecast using an ARIMA model that considers information on detected anomalies
    """
    pipeline = Pipeline(name='detected_anomalies_arima', path=os.path.join('run', hparams.name, 'forecasting',
                                                                           'detected_anomalies_arima'))

    #####
    # Scale load
    ###
    load_scaler = SKLearnWrapper(module=StandardScaler(), name='load_scaler')
    load_scaled = load_scaler(x=pipeline['y_hat'])
    load_scaled_right_shape = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x.values), -1)), x))(
        x=load_scaled)

    #####
    # Shift load by one to exclude actual value from features (i.e. shift value we want to predict out of the data)
    ###
    # NOTE: Shifting will add zeros to the end or the beginning of the data
    load_shifted1 = ClockShift(lag=4, name='clockShift4')(x=load_scaled_right_shape)
    load_shifted24 = ClockShift(lag=96, name='clockShift96')(x=load_scaled_right_shape)
    load_shifted48 = ClockShift(lag=192, name='clockShift192')(x=load_scaled_right_shape)
    load_shifted168 = ClockShift(lag=672, name='clockShift672')(x=load_scaled_right_shape)

    #####
    # Create calendar features
    ###
    calendar_features = CalendarExtraction(country='Germany',
                                           features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                                     CalendarFeature.day_sine, CalendarFeature.monday,
                                                     CalendarFeature.tuesday,
                                                     CalendarFeature.wednesday, CalendarFeature.thursday,
                                                     CalendarFeature.friday, CalendarFeature.hour_cos,
                                                     CalendarFeature.day_cos,
                                                     CalendarFeature.month_cos, CalendarFeature.saturday,
                                                     CalendarFeature.sunday,
                                                     CalendarFeature.workday])(x=pipeline['y_hat'])
    calendar_features_right_shape = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x), -1)), x))(
        x=calendar_features)

    #####
    # Shift detected anomalies
    ###
    # anomaly_shifted = ClockShift(lag=1, name='anomaly_shifted')(x=anomaly_encoded)
    anomaly_shifted1 = ClockShift(lag=1, name='anomaly_shift4')(x=pipeline['anomalies_hat'])
    anomaly_shifted24 = ClockShift(lag=96, name='anomaly_shift96')(x=pipeline['anomalies_hat'])
    anomaly_shifted48 = ClockShift(lag=192, name='anomaly_shift192')(x=pipeline['anomalies_hat'])
    anomaly_shifted168 = ClockShift(lag=672, name='anomaly_shift672')(x=pipeline['anomalies_hat'])
    anomaly_shifted_right_shape1 = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x.values), -1)), x))(
        x=anomaly_shifted1)
    anomaly_shifted_right_shape24 = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x.values), -1)), x))(
        x=anomaly_shifted24)
    anomaly_shifted_right_shape48 = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x.values), -1)), x))(
        x=anomaly_shifted48)
    anomaly_shifted_right_shape168 = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((len(x.values), -1)), x))(
        x=anomaly_shifted168)

    #####
    #  forecasting model
    ###
    forecast = SmTimeSeriesModelWrapper(
        module=ARIMA,
        name='forecast',
        use_exog=True,
        module_kwargs={
            "order": (1, 0, 1)
        }
    )(
        # lags used as exogenous variables required to provide time information
        load_lag1=load_shifted1,
        load_lag2=load_shifted24,
        load_lag3=load_shifted48,
        load_lag4=load_shifted168,
        calendar_features=calendar_features_right_shape,
        lag_anomaly1=anomaly_shifted_right_shape1,
        lag_anomaly2=anomaly_shifted_right_shape24,
        lag_anomaly3=anomaly_shifted_right_shape48,
        lag_anomaly4=anomaly_shifted_right_shape168,
        target=load_scaled
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
            x.values.flatten(),
            dims=['index'],
            coords={'index': x.index.values}
        )
    }, name='forecast')(x=inverse_scaler)

    return pipeline
