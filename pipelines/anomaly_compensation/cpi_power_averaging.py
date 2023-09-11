"""
Assumption: training data doesn't contain any nan-values
This method determines average power values per day per timestamp (averaged over all given weeks) in the training
(fit_method of FunctionModule) to fill nan values in test step. With the filled in power values we can cumulate to
get the energy time series, set missing values again to nan and apply cpi. Thus, the filled in values serve merely as
estimate for the amount of energy that is missing and the concrete assignment of energy to missing timestamps is done
via the Copy paste imputation

after training this should be 7 x (24 / resolution) arrays containing at (i,j) the mean or standard deviation of
day i at time j (thereby time is discretized by the used resolution)
indices are specified with dt.dayofweek for row indices (whatever has number 0 is in the first row...) and for
simplicity distance in 24 / resolution hour steps from 00:00 (i.e. if resolution is 0.25 then 00:00 has index 0,
00:15 has index 1, 00:30 has index 2, ...)
"""

import os

import numpy as np
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules.wrappers.function_module import FunctionModule

from pipelines.anomaly_compensation.copy_paste_imputation import apply_cpi


means_arr = None
stds_arr = None


def get_time_indices(time, ind, resolution):
    '''
    Calculates indices of timestamps.
    '''
    return np.array((time.dt.hour.values[ind] + time.dt.minute.values[ind] / 60 +
                     time.dt.second.values[ind] / 3600) / resolution, dtype=int)


def get_means_stds(y, resolution):
    '''
    Determines mean and standard deviations of power values for each day and time.
    To fill missing data in the power time series.
    '''
    # values per day
    vpd = int(24 / resolution)
    # values per week
    vpw = vpd * 7
    # reshape: want 2-dim array for every week (axis 0 specifies week), power values of each day by row (axis 1 day),
    # and column indicating the timestamp (axis 2 for time) [reshape fills array row-wise, so it should work perfectly]
    # fill with nans (to be able to reshape) and do nanmean and nanstd to ignore added nans
    fill_up = np.repeat(np.nan, vpd * 7 - y.values.size % vpw)
    order_power = np.concatenate((y.values, fill_up)).reshape(-1, 7, vpd)
    if order_power.shape[0] > 1:
        mean_energy_per_ts = np.nanmean(order_power, axis=0)
        std_energy_per_ts = np.nanstd(order_power, axis=0)
    else:
        mean_energy_per_ts = order_power
        std_energy_per_ts = np.repeat(np.nan, vpw).reshape(7, vpd)
    global means_arr
    global stds_arr
    day_indices = y.index.dt.dayofweek.values[0 + vpd * np.arange(0, 7)]
    time_indices = get_time_indices(y.index, np.arange(0, vpd), resolution)
    # rearrange values so that indices are correct (a bit complicated, but there were some overwrite errors)
    # (otherwise the first day appearing in the data comes first, we want day 0 to be first)
    empty_arr_1 = np.repeat(0.0, vpw).reshape(7, vpd)
    empty_arr_2 = np.repeat(0.0, vpw).reshape(7, vpd)

    # sort rows
    empty_arr_1[day_indices, :] = mean_energy_per_ts
    # sort columns
    empty_arr_2[:, time_indices] = empty_arr_1
    means_arr = empty_arr_2.copy()

    # sort rows
    empty_arr_1[day_indices, :] = std_energy_per_ts
    # sort columns
    empty_arr_2[:, time_indices] = empty_arr_1
    stds_arr = empty_arr_2.copy()


def impute_with_cpi(y, resolution):
    '''
    This method:
        For NaN values and missing timestamps in power time series, mean power values are pasted.
        Integrates power time series to get energy time series, by calculating the cumulative sum. And pastes nan values
        back to their original positions (we only used the mean power values to get an estimate for missing energy)
        Applies the CPI method. (CPI takes care about the concrete distribution of energy in gaps of missing values)
        Differentiate discrete to get the power time series back and returns it.
    '''
    # saves positions of nans
    remember_nans = np.isnan(y.values)
    week_indices = y.index.dt.dayofweek.values[remember_nans]
    time_indices = get_time_indices(y.index, remember_nans, resolution)

    # for every nan value look up the respective mean value determined in training (by dayofweek and time)
    imputed_power = y.values
    imputed_power[remember_nans] = means_arr[(week_indices, time_indices)]

    # gets energy time series by calculating the cumulative sum of the power time series
    # and inserts nans again
    energy_nan = np.where(remember_nans, np.nan, np.cumsum(imputed_power))

    # apply cpi for energy time series
    impute_energy = apply_cpi(xr.DataArray(energy_nan, dims="index", coords={"index": y.index}), resolution).values

    # differentiate discrete to get the power time series back and return it
    lagged_energy = np.roll(impute_energy, 1)
    lagged_energy[0] = 0

    return xr.DataArray(impute_energy - lagged_energy, dims="index", coords={"index": y.index})


def compensate(hparams):
    """
    Pipeline to compensate anomalies by CPI
    Demands power time series as input.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'compensation', 'cpi_power_averaging'))

    res = hparams.resolution

    FunctionModule(lambda y: impute_with_cpi(y, res), fit_method=lambda y: get_means_stds(y, res), name='y_hat_comp')(
        y=pipeline['y_hat_comp'])
    return pipeline
