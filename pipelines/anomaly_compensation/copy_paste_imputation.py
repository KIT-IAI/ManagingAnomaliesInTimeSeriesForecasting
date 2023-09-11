import os

import pandas as pd
import numpy as np
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules.wrappers.function_module import FunctionModule

from cpiets.cpi import CopyPasteImputation
from cpiets.utils import estimate_starting_energy


def prepare_imputation(index, is_start, values, timestamps, resolution):
    """
    Prepares data for imputation.
    a. If the partial day doesn't contain any
       nan values (no imputation necessary),
       discard start / end.
    b. If the partial day contains nan values
       in the start / end (imputation necessary),
       fill start / end with nans.
    """
    min_per_ts = np.int64(resolution * 60)
    # indices of the boundary region
    if (is_start and index == 0) or (not is_start and index == timestamps.size - 24 / resolution):
        # start or end is complete, do nothing
        # impute boundary will be false
        # and just return what was given
        boundary = np.array([], dtype=int)
        if not is_start:
            # set index to size, so that nothing
            # is done and everything is returned
            index = timestamps.size
    else:
        boundary = np.arange(0, index) if is_start else np.arange(index, timestamps.size)

    impute_boundary = np.any(np.isnan(values[boundary]))
    if impute_boundary:
        # boundary has to be imputed as well
        if is_start:
            size = (timestamps.dt.hour.values[0]
                    * 60 + timestamps.dt.minute.values[0]) / min_per_ts - 1
            times = np.concatenate(
                (timestamps.values[0] - np.flip(np.arange(1, size + 1))
                 * np.timedelta64(min_per_ts, 'm'),
                timestamps))
            vals = np.concatenate((np.repeat(np.nan, size), values))
            out = size
        else:
            size = (24 * 60 - timestamps.dt.hour.values[-1]
                    * 60 - timestamps.dt.minute.values[-1]) / min_per_ts
            # timestamp with 0 hours and 0 minutes
            # is last entry for each day => got to size + 1
            times = np.concatenate((timestamps,
                                   timestamps.values[-1] + np.arange(1, size + 1)
                                   * np.timedelta64(min_per_ts, 'm')))
            vals = np.concatenate((values, np.repeat(np.nan, size)))
            out = size
    else:
        # store and remove boundary
        if is_start:
            times = timestamps[index:]
            vals = values[index:]
            out = pd.DataFrame({"timestamps": timestamps[:index], "values": values[:index]})
        else:
            times = timestamps[:index]
            vals = values[:index]
            out = pd.DataFrame({"timestamps": timestamps[index:], "values": values[index:]})
    # Attention out can be an index (then start or end had to be imputed)
    # or a pd.DataFrame (then it was discarded and
    # that's why is is stored separately in out)
    return xr.DataArray(times, dims="index", coords={"index": times}), vals, out


def apply_cpi(x, resolution):
    """
    Since CPI is only working for data comprising integer number of days,
    the following adaptation was made in order to make CPI also usable
    if data starting or ending with a partial day:
    a. If the partial day doesn't contain any nan values (no imputation necessary),
       discard start / end. After applying CPI, add the temporarily removed start / end
       again to the imputed values.
    b. If the partial day contains nan values in the start / end (imputation necessary),
       fill start / end with nans. After applying CPI, discard the data conforming to
       the nan values used as padding.
    Checking and filling on nan values does the prepare_imputation method.
    """
    first_minutes = resolution * 60
    days_start = np.argwhere((x.index.dt.hour.values == (first_minutes // 60)) &
                             (x.index.dt.minute.values == (first_minutes % 60)))
    # index of first timestamp of first and last complete day
    first_cmpl_day = days_start[0][0]
    last_cmpl_day = days_start[-1][0]

    # process first the end, then the start,
    # since then indices are still valid for processing start
    timestamps, values, out_end = prepare_imputation(
        last_cmpl_day, False, x.values, x.index, resolution)
    timestamps, values, out_start = prepare_imputation(
        first_cmpl_day, True, values, timestamps, resolution)

    # values per day
    vpd = int(24 / resolution)
    # estimate starting energy
    starting_energy = estimate_starting_energy(pd.Series(values))

    cpi = CopyPasteImputation()
    # cpi expects DataFrame with time column being strings
    cpi.fit(ets=pd.DataFrame({
                "time": np.datetime_as_string(timestamps.values),
                "energy": values}),
            values_per_day=vpd,
            starting_energy=starting_energy)
    imputed_values = cpi.impute().values
    if np.isscalar(out_end):
        # remove last out_end values
        return_values = imputed_values[:(imputed_values.size - int(out_end))]
    else:
        return_values = np.concatenate((imputed_values, out_end['values'].values))
    if np.isscalar(out_start):
        # remove first out_start values
        return_values = return_values[int(out_start):]
    else:
        return_values = np.concatenate((out_start['values'].values, return_values))

    return xr.DataArray(return_values, dims="index", coords={"index": x.index})


def compensate(hparams):
    """
    Pipeline to compensate anomalies by CPI.
    Demands energy time series as input.
    Assumption:
    (a) resolution * 60 (number of minutes between two timestamps)
        is an integer number
    (b) first timestamp of a day has time 00:00 + resolution
        and last timestamp has time 00:00, i.e., if 15 min are
        between two timestamps first timestamps of days
        have time 00:15 and last timestamps of days
        have time 00:00
    """
    pipeline = Pipeline(path=os.path.join(
        'run', hparams.name, 'compensation', 'copy_paste_imputation'))

    FunctionModule(lambda x: apply_cpi(x, hparams.resolution), name='y_hat_comp')(
        x=pipeline['y_hat_comp'])
    return pipeline
