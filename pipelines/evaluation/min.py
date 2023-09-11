import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.min_summary import MinErr


def eval(hparams):
    """
    Determine the minimum error for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'min'))

    #####
    # calculate min
    ###
    MinErr(name='minerr')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
