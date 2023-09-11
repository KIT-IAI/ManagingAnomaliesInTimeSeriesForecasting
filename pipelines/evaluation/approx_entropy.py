import os

import numpy as np

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.metric_base import MetricBase


class ApproxEntropy(MetricBase):
    """
    approximate entropy summary module
    """

    def _apply_metric(self, p, t):
        result = approximate_entropy(p)
        return result


def approximate_entropy(data):
    """
    Calculate the approximate entropy as defined in https://en.wikipedia.org/wiki/Approximate_entropy. It can be used to
    assess the regularity of time series (Wikipedia refers to https://www.mdpi.com/1099-4300/21/6/541/htm)

    Eventually, we subtract the entropy of the m-slices from the (m+1)-slices of the time series, which is a measure for
    the uncertainty of the m+1 element succeeding a fix m-block
    """
    # standardize data
    x = (data - np.mean(data)) / np.std(data)
    # parameters of approximate entropy
    m = 3  # length of slices that are compared
    r = 0.1 * np.std(x)  # filter level
    n = len(x)

    # in order to calc the entropy faster, we unfold the 1d data into a 2d array, whereby the i-th column should
    # contain the vector data shifted by i and the end is filled with NANs. Thus, every row of unfold contains slices
    # of length m:  unfold[l,:] = data[l:l+m+1] = (data[l], data[l+1], data[l+2], ..., data[l+m])
    unfold = np.full((n, m+1), np.nan)
    for i in range(0, m+1):
        unfold[0:n-i, i] = x[i:n]

    def find_matches(row):
        # find all rows s (i.e. all slices x[j:j+m]) whose max pointwise difference between row and s
        # is less than r (do that by checking that every pointwise difference is less than r)
        m_matches = np.all(np.abs(row[0:m] - unfold[:, 0:m]) <= r, axis=1)
        # if also the last pointwise difference is less than r, it's a m+1 match
        m_plus_one_matches = np.all(np.array([m_matches, (np.abs(row[m] - unfold[:, m]) <= r)]), axis=0)
        # return number of true values in the respective arrays => sum it
        return np.array([m_matches.sum(), m_plus_one_matches.sum()])

    num_of_slices = np.apply_along_axis(find_matches, 1, unfold)
    # replace 0 to apply log(num_of_slices / np.array([n-m+1, n-m]) by n-m+1 (1st column) or n-m (2nd column)
    # => we neglect these rows since log(1) = 0 and we sum over the elements
    num_of_slices = np.where(num_of_slices != 0, num_of_slices, np.array([n-m+1, n-m]))

    phi_m, phi_m_plus_one = np.sum(np.log(num_of_slices / np.array([n-m+1, n-m])), axis=0)
    result = phi_m / (n - m + 1) - phi_m_plus_one / (n - m)
    return result


def eval(hparams):
    """
    Determine the approximate entropy for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'approxEntropy'))

    #####
    # calculate approx entropy
    ###
    ApproxEntropy(name='approxEntropy')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
