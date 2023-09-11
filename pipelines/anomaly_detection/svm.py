import os

from pyod.models.ocsvm import OCSVM

from sklearn.preprocessing import MinMaxScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, FunctionModule, SKLearnWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray

global svm

def detect(hparams, lag_features=6):
    """
    Detect anomalies with one-class SVM
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
    global svm
    anomalies = svm.predict(x.values.reshape(-1,1))
    return numpy_to_xarray(anomalies, x)


def fit_model(x):
    global svm
    # Contamination = the proportion (percentage) of outliers in the data set
    svm = OCSVM(contamination=0.05)
    svm.fit(x.values.reshape(-1,1))
