import os
import numpy as np

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


def detect(hparams):
    """  Identity detetion method that detects no anomalies. """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'identity'))

    FunctionModule(lambda x: x, name='y')(x=pipeline['y'])
    FunctionModule(lambda x: x, name='y_hat')(x=pipeline['y_hat'])
    FunctionModule(lambda x: x, name='anomalies')(x=pipeline['anomalies'])
    FunctionModule(lambda y: numpy_to_xarray(np.zeros(y.shape), y),
                   name='anomalies_hat')(y=pipeline['y_hat'])

    return pipeline
