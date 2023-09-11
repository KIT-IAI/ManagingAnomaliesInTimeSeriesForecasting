import logging

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

## On a Windows PC of IAI, the default backend of matplotlib was 'tkinter' which resulted in a data leakage
# in the grid_search (the backend 'agg' solves this issue (#74))
matplotlib.use('agg')


def anomaly_plot(ts, anomalies, anomalies_hat, name='anomaly_plot.png'):
    """
    Plot results of anomaly detection, i.e. FN, TP, FP
    """
    # get boolean values of anomaly positions
    anomalies = anomalies.values != 0
    anomalies_hat = anomalies_hat.values != 0

    # plot actual load curve
    plt.plot(ts.index, ts)

    # plot FN, TP and FP on load values
    plt.scatter(ts.index[anomalies & ~anomalies_hat].values, ts[anomalies & ~anomalies_hat].values,
                color='r', marker='x', label='False Negative')
    plt.scatter(ts.index[anomalies & anomalies_hat].values, ts[anomalies & anomalies_hat].values,
                color='g', marker='x', label='True Positive')
    plt.scatter(ts.index[~anomalies & anomalies_hat].values, ts[~anomalies & anomalies_hat].values,
                color='y', marker='x', label='False Positive')

    # set some plot parameters
    plt.xlabel('Time [h]')
    plt.xticks(rotation=45)
    plt.ylabel('Load [MW]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def data_plots(y, y_hat=None, name='data_plot.png', labels=['in', 'out'], loc='lower right'):
    """
    Plot time series
    """
    plt.plot(y.index, y, label=labels[0], linewidth=1, alpha=0.75, color="blue")

    if y_hat is not None:
        plt.plot(y.index, y_hat, label=labels[1], linewidth=1, alpha=0.75, color="red")

    maximum = y.values.max() if y_hat is None else np.maximum(y.values.max(), y_hat.values.max())
    # TODO: max
    #plt.ylim(0.9 * y.min(), 1.1 * maximum)
    plt.xlabel('Time [h]')
    plt.ylabel('Load [MW]')
    plt.xticks(rotation=45)
    plt.legend(loc=loc)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def daily_load_plots(y, width, prefix='daily_plot'):
    """
    Plot daily load curves
    """
    # reshape y to have dim (days, width)
    start_index = np.argwhere(
        ((y.index.dt.hour == 0) & (y.index.dt.minute == 0)).values
    )[0][0]
    y = y.values[start_index:]
    y = y[:len(y) - len(y) % width].reshape(-1, width)

    # plot all days
    for day in y:
        plt.plot(np.arange(width) / (width / 24), day, alpha=0.5)
    plt.xlabel('Time [h]')
    plt.ylabel('Load [MW]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{prefix}_24h_all_days.png')
    plt.close()

    # plot daily mean with standard deviation (upper and lower)
    mean = y.mean(axis=0).flatten()
    std = y.std(axis=0).flatten()
    plt.plot(np.arange(width) / (width / 24), mean, color='b', linewidth=2)
    plt.plot(np.arange(width) / (width / 24), mean + std, 'r--')
    plt.plot(np.arange(width) / (width / 24), mean - std, 'r--')

    # set some plot parameters
    plt.xlabel('Time [h]')
    plt.ylabel('Load [MW]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{prefix}_daily_mean_std.png')
    plt.close()


def result_plot(results, eval_methods, forecast_methods, name='evaluation_plot'):
    """
    Plot evaluation of forecast for y, y_hat, and (if available) y_hat_comp
    """
    # We assume that the order of results corresponds to the order of forecast_methods
    # otherwise, results are wrongly labeled in the bar plot
    bar_width = 0.25
    # iterate through eval_methods and plot for each eval method the results of the different methods
    for evaluation in eval_methods:
        # we assume the result.keys() to be of the form method/evaluation/{y,y_hat,y_hat_comp}
        y = [results[i] for i in results.keys() if evaluation == i.split('/')[1] and 'y' == i.split('/')[2]]
        y_hat = [results[i] for i in results.keys() if evaluation == i.split('/')[1] and 'y_hat' == i.split('/')[2]]
        y_hat_comp = [results[i] for i in results.keys() if evaluation == i.split('/')[1]
                      and 'y_hat_comp' == i.split('/')[2]]

        # if results are not scalar, we cannot print a bar plot -> skip it
        if not np.all(list(map(np.isscalar, (y + y_hat + y_hat_comp)))):
            continue

        bar_left = np.arange(len(forecast_methods))
        bar_mid = bar_left + bar_width
        bar_right = bar_mid + bar_width

        plt.bar(bar_left, y, width=bar_width, label='y', color="orange", edgecolor="darkred")
        if y_hat:
            plt.bar(bar_mid, y_hat, width=bar_width, label='y_hat', color="limegreen", edgecolor="darkgreen")
        if y_hat_comp:
            plt.bar(bar_right, y_hat_comp, width=bar_width, label='y_hat_comp', color='deepskyblue', edgecolor='darkblue')

        plt.xlabel("forecasting methods")
        plt.xticks(bar_mid, forecast_methods)
        plt.title(evaluation)

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(name + '_' + evaluation + '.png')
        plt.close()


def plot_overview(train, test, name='data_overview.png'):
    """
    Plot overview of training data, testing data, y, y_hat, y_hat_comp
    """
    # plot 6 graphs together, with no space in between and joint axes
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
    axis = gs.subplots(sharex='col', sharey='row')

    axis[0, 0].set_title("training data")
    axis[0, 1].set_title("testing data")
    axis[0, 0].set_ylabel('y')
    axis[1, 0].set_ylabel('y_hat')
    axis[2, 0].set_ylabel('y_hat_comp')
    # actually, i just want to specify rotation=45. However since then I need to specify the tick labels, I also need
    # to specify the tick position. Otherwise a warning is printed
    n = train['y'].index.size
    ticks = train['y'].index[np.arange(0, n, int(n/7))].values
    axis[2, 0].set_xticks(ticks)
    axis[2, 0].set_xticklabels(ticks, rotation=45)
    n = test['y'].index.size
    ticks = test['y'].index[np.arange(0, n, int(n / 7))].values
    axis[2, 1].set_xticks(ticks)
    axis[2, 1].set_xticklabels(ticks, rotation=45)

    axis[0, 0].plot(train['y'].index, train['y'], label='y', color="orange")
    axis[0, 1].plot(test['y'].index, test['y'], label='y', color="orange")
    axis[1, 0].plot(train['y_hat'].index, train['y_hat'], label='y_hat', color="limegreen")
    axis[1, 1].plot(test['y_hat'].index, test['y_hat'], label='y_hat', color="limegreen")
    axis[2, 0].plot(train['y_hat_comp'].index, train['y_hat_comp'], label='y_hat_comp', color="deepskyblue")
    axis[2, 1].plot(test['y_hat_comp'].index, test['y_hat_comp'], label='y_hat_comp', color="deepskyblue")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def correlation_plot(directory_name, corr, resolution, suffix='', cross=True, lines=True):
    if cross:
        prefix = 'cross'
        lags = np.arange(-(len(corr['y']) - 1) / 2, (len(corr['y']) + 1) / 2)    # assume lags to be symmetric around 0
    else:
        prefix = 'auto'
        lags = np.arange(0, len(corr['ground_truth']))  # assume lags to be 0, 1, 2, ..., size

    # plot 3 graphs together, with no space in between and joint axes
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    axis = gs.subplots(sharex='col', sharey='row')

    axis[0].set_title(suffix + " " + prefix + "-correlation")
    multiple = np.maximum(int(lags.size / (24 / resolution) / 15), 1)   # we do not want to have more than 16 ticks
    xticks = lags[lags % (multiple * (24 / resolution)) == 0]     # do ticks in multiple * days
    axis[2].set_xticks(xticks)
    axis[2].set_xticklabels(xticks / (24 / resolution))
    axis[2].set_xlabel('lag [days]')

    k = 0

    for key in corr.keys():         # iterate through the rows
        if key == 'ground_truth':
            continue
        axis[k].set_ylabel(key)
        axis[k].set_ylim([-1.1, 1.1])
        yticks = [-1, -0.5, 0, 0.5, 1]
        axis[k].set_yticks(yticks)
        # draw gray grid
        for x in xticks:
            axis[k].axvline(x=x, color='lightgray', linestyle='--')
        for y in yticks:
            axis[k].axhline(y=y, color='lightgray', linestyle='--')

        if lines:
            label = 'y_* with ground truth'
            if not cross:       # in cross-correlation, we don't have ground_truth
                axis[k].plot(lags, corr['ground_truth'], color='blue', label='ground_truth')
                label = 'y_*'
            axis[k].plot(lags, corr[key], color='orange', label=label)
        else:
            axis[k].bar(lags, corr[key], width=1, color='orange', label='y_*')
            if not cross:       # in cross-correlation, we don't have ground_truth
                axis[k].bar(lags, corr['ground_truth'], width=1, color='blue', label='ground_truth')
                # redraw bars that cover each other
                redraw = np.where(np.sign(corr['ground_truth']) == np.sign(corr[key]),
                                  np.sign(corr[key]) * np.minimum(np.abs(corr['ground_truth']), np.abs(corr[key])), 0)
                axis[k].bar(lags, redraw, width=1, color='brown', label='intersection')
        k = k + 1

    axis[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(directory_name + '/' + prefix + '_correlation_plot_' + suffix + '.png')
    plt.close()


def correlation_histograms(corr, cross=True, forecast_model='', name='corr_hist.png', origin='test',
                           cross_origin='test'):
    plot_corr = corr.copy()
    for key in corr.keys():
        if np.all(np.isnan(corr[key])):
            plot_corr.pop(key)           # remove key-value-pairs consisting of only nans
    if len(plot_corr) == 0:
        return          # stop if nothing is plotted
    # plot graphs together with identically scaled x-axis
    nr = len(plot_corr.keys())
    grid = (2, int(round(nr / 2))) if nr > 3 else (1, nr)
    index_list = list(product([0, 1], repeat=2)) if nr > 3 else np.arange(0, nr)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(grid[0], grid[1])
    axis = gs.subplots(sharex='all')
    if nr == 1:
        axis = [axis]          # if we plot only one hist, put it in a list in order to use index 0

    bin_width = 2 / 30         # correlations range from -1 to 1 (width = 2) and we want 30 bins

    if cross:
        title_text = f"Histograms of cross-correlation at all lags between ground_truth ({cross_origin}) and ..."
    else:
        title_text = "Histograms of auto-correlation at all lags"
    title = f"{forecast_model}\n{title_text}" if forecast_model != '' else title_text
    fig.suptitle(title)

    i = 0
    for key in plot_corr.keys():
        axis[index_list[i]].set_title(f"{key} ({origin})")
        # since plots share same x-axis scale, adapt also number of bins, so that bins are almost equally wide
        span = np.nanmax(corr[key]) - np.nanmin(corr[key])
        # span > 2 means corr values don't make sense!
        bin_number = int(np.ceil(span / bin_width)) if (span <= 2 and nr > 1) else None
        if np.isnan(span):
            logger = logging.getLogger("Log")
            logger.addHandler(logging.StreamHandler())
            logger.info(
                "Could not calculate difference between min and max values on input data. "
                "Skipping correlation_histogram plot."
            )
            return
        else:
            axis[index_list[i]].hist(corr[key], bins=bin_number, color="dodgerblue", edgecolor="darkblue")
            i = i + 1

    plt.tight_layout()
    plt.savefig(name)
    plt.close()
