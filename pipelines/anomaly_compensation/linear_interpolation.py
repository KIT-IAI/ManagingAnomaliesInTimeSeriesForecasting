import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import LinearInterpolater


def compensate(hparams):
    """
    Compensate anomalies with a linear interpolation
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'compensation', 'linear_interpolation'))

    #####
    # Apply linear interpolation
    ###
    LinearInterpolater(
        dim='index', name='y_hat_comp', method='nearest'
    )(x=pipeline['y_hat_comp'])

    return pipeline
