import os

import numpy as np

import statsmodels.tsa.statespace.sarimax as tsa

from sklearn.preprocessing import MinMaxScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, FunctionModule, SKLearnWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray

global arima
global stdev

def detect(hparams, lag_features=6):
    """
    Detect anomalies with SARIMAX, a version of ARIMA which considers seasonal changes
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'classification'))

    #####
    # Scale features
    ###
    y_scaler = SKLearnWrapper(module=MinMaxScaler(), name='y_hat')
    y_scaled = y_scaler(x=pipeline['y_hat'])
   
    #####
    # Create features
    ###
    # Shift load to have a load feature to forecast load with a model.
    # NOTE: Shifting will add zeros to the end or the beginning of the data
    features = {}
    for i in range(lag_features):
        y_lag = ClockShift(name=f'y_{str(i)}h', lag=i)(x=y_scaled)
        features[f'f_y{str(i)}'] = y_lag

    FunctionModule(lambda x: determine_anomalies(x),
                   lambda x: fit_model(x),
                   name='anomalies_hat')(x=pipeline['y_hat'])

    return pipeline


def determine_anomalies(x):
    global arima
    global stdev

    data = x.values.flatten()

    prediction = arima.predict(start = 0, end = (len(x) - 1))
    prediction = prediction.flatten()
    
    # Vectorized subtraction of the arrays prediction and data
    difference = abs(prediction - data)
    
    # Error threshold
    threshold = 2*stdev
    
    def scoring_anomaly(x):
        if x >= threshold:
            return 1
        else:
            return 0
   
    vector_operation = np.vectorize(scoring_anomaly)    
    
    anomalies = vector_operation(difference)

    return numpy_to_xarray(anomalies, x)


def fit_model(x):
    global arima
    global stdev
    
    # The seasonal order value of 24 signifies the daily 24 hour cycle
    placeholder = tsa.SARIMAX(endog = x.values.reshape(-1,1), order = (1,0,0), seasonal_order = (1, 0, 0, 24))
    
    # Standard deviation, which is used as error threshold
    stdev = np.std(x.values)
    arima = placeholder.fit()
