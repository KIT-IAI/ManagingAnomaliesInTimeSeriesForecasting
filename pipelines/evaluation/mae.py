import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.mae_summary import MAE


def eval(hparams):
    """
    Determine the MAE for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'MAE'))

    #####
    # calculate MAE
    ###
    MAE(name='mae')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
