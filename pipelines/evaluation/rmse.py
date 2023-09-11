import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.rmse_summary import RMSE


def eval(hparams):
    """
    Determine the RMSE for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'RMSE'))

    #####
    # calculate RMSE
    ###
    RMSE(name='rmse')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )
    return pipeline
