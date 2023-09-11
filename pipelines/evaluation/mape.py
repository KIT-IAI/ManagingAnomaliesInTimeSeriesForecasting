import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.mape_summary import MAPE


def eval(hparams):
    """
    Determine the MAPE for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'MAPE'))

    #####
    # calculate MAPE
    ###
    MAPE(name='mape')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
