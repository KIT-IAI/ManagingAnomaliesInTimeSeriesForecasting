import os

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import ClockShift, FunctionModule, SKLearnWrapper
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray

def detect(hparams, lag_features=6):
    """
    Detect anomalies with k-means
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
    # Number of clusters. 3 was best after using the elbow method on the data
    k_clusters = 3

    kmeans = KMeans(n_clusters = k_clusters)

    kmeans.fit(x.values.reshape(-1,1))

    cc = kmeans.cluster_centers_
    
    # Initialize the arrays to any value so that they can be assigned to. 
    closest = x.values.reshape(-1, 1)
    anomalies = x.values.reshape(-1, 1)
    
    # Standard deviation of the time series
    stdev = np.std(x.values)
    
    # Threshold of the difference between the data point and closest cluster centroid
    threshold = 2 * stdev

    # Finds closest cluster centroid and calculates distances to it
    for i in range(0, (len(x) - 1)):
        closest = find_nearest(cc, x.values.reshape(-1, 1)[i])
        difference = abs(x.values.reshape(-1, 1)[i] - closest)

        # If the distance is greater than the threshold, it is marked as an anomaly
        if (threshold < difference):
            anomalies[i] = 1
        else:
            anomalies[i] = 0

    anomalies = anomalies.flatten()

    return numpy_to_xarray(anomalies, x)

def find_nearest(array, value):
    # Finds closest entry in array to the value and returns it
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
