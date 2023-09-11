import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule


def forecast(hparams):
    """
    Calculates a forecast using the current value
    """
    pipeline = Pipeline(name='current_value', path=os.path.join('run', hparams.name, 'forecasting', 'current_value'))

    #####
    # Apply forecasting model
    ###
    y_hat = FunctionModule(lambda x: x, name='forecast')(x=pipeline['y_target'])

    return pipeline
