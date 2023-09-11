import os

import xarray as xr

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, FunctionModule, SKLearnWrapper


def detect(hparams, lag_features=6):
    """
    Detect anomalies with a random forest classification
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'random_forest'))

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

    #####
    # Define classification
    ###
    classification = SKLearnWrapper(RandomForestClassifier(), 'anomalies_hat')(
            **features, target=pipeline['anomalies']
    )
    FunctionModule(lambda x: {
            'anomalies_hat': xr.DataArray(x.values.flatten(), dims=['index'], coords=[x.index])
        }, name='anomalies_hat')(x=classification)

    return pipeline
