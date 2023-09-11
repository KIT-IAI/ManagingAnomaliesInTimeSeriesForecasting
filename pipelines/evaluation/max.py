import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.max_summary import MaxErr


def eval(hparams):
    """
    Determine the maximum error for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'max'))

    #####
    # calculate max
    ###
    MaxErr(name='maxerr')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
