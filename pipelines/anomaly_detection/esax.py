import os

import numpy as np
import pandas as pd
import xarray as xr

from statistics import mean, median, stdev
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
import esax.get_subsequences as subsequences
import esax.get_motif as motifs
import esax.plots as plots
import modules.esax_wrapper as e_wrapper

global repr_motif
global std_seq
global std_seq_ts
global motif_raw
global confidence_band
global results
global run_id


def eSAX(x: xr.DataArray, resolution):
    filepath = os.path.join('run', run_id)

    # Get subsequences
    ts_subs, startpoints, indexes_subs = subsequences.get_subsequences(x, resolution)

    # Get motifs
    if ts_subs:
        found_motifs = motifs.get_motifs(x, ts_subs, breaks=10, word_length=0, num_iterations=0, mask_size=2,
                                         mdr=2.5, cr1=5.0, cr2=1.5)
        if found_motifs:
            plots.plot_ecdf(found_motifs['ecdf'], filepath)
            parsed_indexes = pd.to_datetime(x.index)
            plots.plot_motifs(parsed_indexes, found_motifs['motifs_raw'], found_motifs['indexes'], filepath)
            plots.plot_repr_motif(found_motifs['motifs_raw'], filepath)

    found_motifs = e_wrapper.prepare_results(found_motifs, startpoints, indexes_subs)
    return found_motifs


def locate_anomaly_points_with_quantiles(x):
    """
    Detect anomalous points based on a high and low quantile, which are derived from the group
    of sequences in one motif. Points which exceed this interval are marked as anomaly.
    @param x: training/test data
    @return: binary representation of the training/test data (0: no anomaly, 1: anomaly)
    """

    if confidence_band is not None:
        lower_bnd_ts = eliminate_time_shift(confidence_band[0], x)
        upper_bnd_ts = eliminate_time_shift(confidence_band[1], x)
        anomalies = np.array([], dtype=np.int64)
        indexes = x.values < lower_bnd_ts.values
        indexes |= x.values > upper_bnd_ts.values
        indexes = indexes.astype(int)
        anomalies = np.concatenate((anomalies, indexes), axis=None)

    return anomalies


def locate_anomaly_points_naive(x, selected_range=2):
    """
    Determine anomalous points based on the values of the representative daily motif and a threshold
    value. If the point in the training/test data is not in the interval, it is marked as anomaly.
    @param selected_range: custom range for anomaly detection (same as in statistical)
    @param x: training/test data
    @return: binary representation of the training/test data (0: no anomaly, 1: anomaly)
    """
    # this method is not used at the moment
    repr_motif_ts = eliminate_time_shift(repr_motif, x)
    std_seq_ts = eliminate_time_shift(std_seq, x)

    anomalies = np.zeros(len(x))
    for i in range(0, len(x)):
        if x.values[i] > repr_motif_ts[i] + selected_range * std_seq_ts[i]:
            anomalies[i] = 1
        elif x.values[i] < repr_motif_ts[i] - selected_range * std_seq_ts[i]:
            anomalies[i] = 1

    return anomalies


def locate_anomaly_days(ts_subs, x, selected_range=1):
    """
    Determine whole days as anomalies. If the dtw distance or the euclidean distance between a day in training/test data
    and the representative daily motif is higher than a threshold, the whole day is marked as an anomaly.
    @param selected_range: custom range for anomaly detection (same as in statistical)
    @param x: training/test data
    @param ts_subs: training/test data divided into daily subsequences
    @return: binary representation of the training/test data (0: no anomaly, 1: anomaly)
    """
    anomalies_indices = np.zeros(len(x))
    for idx, sequence in enumerate(ts_subs):
        if mean(sequence) > mean(repr_motif.values) + selected_range * stdev(repr_motif.values):
            anomalies_indices[idx * len(sequence):(idx * len(sequence) + len(sequence))] = 1
        elif mean(sequence) < mean(repr_motif.values) - selected_range * stdev(repr_motif.values):
            anomalies_indices[idx * len(sequence):(idx * len(sequence) + len(sequence))] = 1

    return anomalies_indices


def eliminate_time_shift(seq, x):
    """
    Eliminate the time shift between the representative daily motif and the training/test data,
    which can start at an arbitrary point.
    @param seq: representative daily motif
    @param x: training/test data
    @return: the representative daily motif repeated so it has the same length like training/test data
                and starts at the same point of time
    """
    date_filter = pd.DatetimeIndex(x.index)[0].strftime('%H:%M:%S')
    seq_idx = pd.DatetimeIndex(seq.index).strftime('%H:%M:%S')
    time_offset_idx = np.where(seq_idx == date_filter)[0]
    # alignment of the two parts, so that the comparison time series starts at the right index
    seq_start = seq[time_offset_idx[0]:]

    full_reps = int((len(x) - len(seq_start)) / len(seq))
    lenght_fullreps = len(seq) * full_reps + len(seq_start)
    rest_length = len(x) - lenght_fullreps

    seq_ts = seq_start
    # repeat the standard pattern to a time series as long as the training-/test data
    for i in range(int(full_reps)):
        seq_ts = xr.concat((seq_ts, seq), dim='index')

    # append the part shorter than the standard pattern
    seq_ts = xr.concat((seq_ts, seq[:rest_length]), dim='index')

    return seq_ts


def determine_motif(x, measuring_interval: float, quantile_lower: float, quantile_upper: float):
    """ Determine the representative motif given the output of eSAX
    At the moment, all subsequences in each motif are condensed to their mean --> repr_motif
    NOTE: Only the first one of the representative motifs is used"""

    global repr_motif, std_seq, motif_raw, confidence_band, results
    results = eSAX(x, measuring_interval)
    if results is None:
        return

    # in case eSAX does not find any motifs, an exception is thrown and nothing is returned
    motif_raw = results['motifs_raw']
    confidence_band = determine_confidence_interval(quantile_lower, quantile_upper)
    motif_indices = results['motifs_raw'].get_index('motif').values
    nr_of_motifs = max(motif_indices) + 1

    # mean of all sequences in one moment at each point of time
    repr = []
    # std of all measurements at each point of time
    std_seq = []

    motif_raw_first = motif_raw[motif_raw.coords['motif'] == 0]
    for j in range(0, motif_raw_first.shape[1]):
        repr.append(median(motif_raw_first[:, j].values))
        std_seq.append(stdev(motif_raw_first[:, j].values))
    repr_motif = xr.DataArray(repr, coords={'index': ('index', results['ts_subs'][0].index.values)}, dims=['index'])
    std_seq = xr.DataArray(std_seq, coords={'index': ('index', results['ts_subs'][0].index.values)}, dims=['index'])

    return repr_motif


def determine_anomalies(x, measuring_interval: float,
                        anomaly_resolution: str = 'point', anomaly_identification: str = 'threshold'):
    """
    Determine anomalies given the representative motif
    @param x: training/test data
    @param measuring_interval: float variable for measurements per hour
    @param anomaly_resolution: string ('point' or 'day')
    @param anomaly_identification: string ('threshold' or 'quantiles')
    @return: binary representation of the training/test data (0: no anomaly, 1: anomaly)
    """

    global repr_motif, results

    if results is None:
        return xr.DataArray(np.zeros(len(x), dtype=np.int64),
                            coords={'index': ('index', x.index.values)}, dims=['index'])

    subs, _, _ = subsequences.determine_subsequences(x, 'none', round(24 / measuring_interval))

    if anomaly_resolution == 'point':
        if anomaly_identification == 'quantiles':
            indices = locate_anomaly_points_with_quantiles(x)
        elif anomaly_identification == 'threshold':
            indices = locate_anomaly_points_naive(x)
        else:
            indices = np.zeros(len(x))
    elif anomaly_resolution == 'day':
        indices = locate_anomaly_days(subs, x)
    else:
        indices = np.zeros(len(x))

    # mark the anomalies as 1 in the anomalies data
    anomalies = xr.DataArray(indices, coords={'index': ('index', x.index.values)}, dims=['index'])

    return anomalies


def determine_confidence_interval(lower_quantile: float, upper_quantile: float):
    """
    Determine two time series with the length of one day, which represent the higher/lower quantile of the sequences
    in one motif for each point.
    @param lower_quantile: float in [0,1]
    @param upper_quantile: float in [0,1]
    @return: two time series that represent the upper/lower quantile band
    """
    global motif_raw
    upper_bnd_ts = xr.DataArray
    lower_bnd_ts = xr.DataArray
    for i in range(0, len(motif_raw[0])):
        upper_bnd_ts = motif_raw.quantile(q=upper_quantile, dim='motif')
        lower_bnd_ts = motif_raw.quantile(q=lower_quantile, dim='motif')

    return lower_bnd_ts, upper_bnd_ts


def detect(hparams):
    global run_id
    """ Find typical motifs in a time series with Energy Time Series Motif Discovery using
    Symbolic Aggregated Approximation (eSAX) """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'detection', 'esax'))
    run_id = hparams.name

    _ = FunctionModule(lambda y, y_hat: determine_anomalies(y_hat, hparams.resolution, hparams.detect_esax_anomaly_resolution,
                                                            hparams.detect_esax_anomaly_identification),
                       lambda y, y_hat: determine_motif(y, hparams.resolution, hparams.detect_esax_quantiles_lower,
                                                        hparams.detect_esax_quantiles_upper),
                       name='anomalies_hat')(
        y=pipeline['y'], y_hat=pipeline['y_hat'])

    return pipeline
