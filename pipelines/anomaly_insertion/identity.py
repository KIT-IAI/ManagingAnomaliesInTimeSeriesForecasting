import os

import numpy as np

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


def insert(hparams):
    """
    Insert no anomalies to pipeline.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'insertion', 'identity'))

    FunctionModule(lambda x: x, name='y')(x=pipeline['y'])
    FunctionModule(lambda x: x, name='y_hat')(x=pipeline['y'])
    FunctionModule(lambda x: numpy_to_xarray(np.zeros(x.shape), x))(x=pipeline['y'])

    return pipeline
