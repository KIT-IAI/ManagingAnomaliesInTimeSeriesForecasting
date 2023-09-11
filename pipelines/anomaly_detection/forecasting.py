import os

import numpy as np
import xarray as xr

from scipy import stats

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts.modules import FunctionModule

import pipelines.forecasting.compensated_anomalies_forecasting as forecasting_methods


def determine_anomalies(y, forecast, selected_range=2):
    """
    Determine anomalies based on a forecast
    """
    # calculate difference between actual value and forecast value
    diff = (forecast.values - y.values) ** 2
    diff = diff.flatten()
    mean = diff.mean()
    iqr = stats.iqr(diff)
    # determine anomalies based on threshold
    anomalies = np.abs(diff) > mean + selected_range * iqr

    # save data for later return
    data = dict()
    data['anomalies_hat'] = xr.DataArray(anomalies, y.coords)

    return data


def detect(hparams):
    """
    Detect anomalies using a forecast
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'forecasting'))

    #####
    # Define forecast
    ###
    forecast_pipeline = getattr(
        forecasting_methods, hparams.detect_forecasting_method
    ).forecast(hparams)

    forecast_pipeline(y_target=pipeline['y'], computation_mode=ComputationMode.Train)
    forecast = forecast_pipeline(y_target=pipeline['y_hat'], computation_mode=ComputationMode.Transform)

    #####
    # Determine anomalies given a forecast
    ###
    FunctionModule(lambda y, y_forecast: determine_anomalies(y, y_forecast),
                   name='anomalies_hat')(y=pipeline['y_hat'], y_forecast=forecast['forecast'])

    return pipeline
