import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


def compensate(hparams):
    """ Pipeline to compensate anomalies by linear interpolation """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'compensation', 'identity'))
    FunctionModule(lambda x: x, name='y_hat_comp')(x=pipeline['y_hat_comp'])
    return pipeline
