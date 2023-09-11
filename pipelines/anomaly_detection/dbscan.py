import os

import numpy as np

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, FunctionModule, SKLearnWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray

def detect(hparams, lag_features=6):
    """
    Detect anomalies with DBSCAN
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
                   name='anomalies_hat')(x=pipeline['y_hat'])

    return pipeline

def determine_anomalies(x):
    # Changes the noise cluster assignment from -1 to 1 and all 
    # other (inlier) clusters to 0
    def adjust_to_pywatts_score(x):
        if x < 0:
            return 1
        else:
            return 0
   
    vector_operation = np.vectorize(adjust_to_pywatts_score)    
    
    # 5% of data is used as min_samples
    minimum_samples = np.ceil(len(x)/20)

    # eps = 20 is chosen after experimentation
    dbscan = DBSCAN(eps = 20, min_samples = minimum_samples)

    anomalies = dbscan.fit_predict(x.values.reshape(-1,1))
    
    anomalies = vector_operation(anomalies)
    
    return numpy_to_xarray(anomalies, x)
  
