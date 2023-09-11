import os
import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules.wrappers import FunctionModule
from prophet import Prophet


m = None


def fit(x):
    """
    Initialize prohpet for later predictions by fitting
    prophet model with power values during training.
    """
    time = x.index
    power = x

    ds = pd.DataFrame({'ds': time.values, 'y': power.values})

    global m
    m = Prophet(changepoint_prior_scale=0.5)
    m.fit(ds)


def predict(x):
    """
    Calculates the compensation via the Prophet method, which uses a
    decomposable time series model with the components trend, seasonality
    and holidays (see "Forecasting at Scale" by Taylor et al. 2018).
    """

    time = x.index
    power = x

    ds = pd.DataFrame({'ds': time.values})
    prediction = power.values.copy()
    global m
    forecast = m.predict(ds)
    nan_idx = np.isnan(prediction)
    prediction[nan_idx] = forecast['yhat'][nan_idx]

    da = xr.DataArray(prediction, dims=['index'], coords=[time.values])
    import pickle
    with open('compensated.pkl', 'wb') as file:
        pickle.dump(da, file)

    return da


def compensate(hparams):
    """
    Compensate anomalies with the Prophet forecasting method.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name,
                                          'compensation', 'prophet'))

    #####
    # Apply Prophet
    ###
    _ = FunctionModule(lambda x: predict(x),
                       lambda x: fit(x),
                       name='y_hat_comp')(x=pipeline['y_hat_comp'])
    return pipeline
