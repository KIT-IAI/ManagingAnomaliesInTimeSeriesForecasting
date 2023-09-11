import os

from utils.plots import anomaly_plot


def output(hparams, train, test):
    """
    Print and plot information after the anomaly detection
    """
    # Print information on detection step
    print('Detection step:')
    print(f'train - {train.keys()}')
    if test is not None:
        print(f'test - {test.keys()}')
    print()

    # Plot FN, TP, FP identified in detection (either for the train or the test set)
    if 'anomalies' in train:
        anomaly_plot(train['y_hat'], train['anomalies'], train['anomalies_hat'],
                     name=os.path.join('run', hparams.name, 'detection_train.png'))

    if test is not None and 'anomalies' in test:
        anomaly_plot(test['y_hat'], test['anomalies'], test['anomalies_hat'],
                     name=os.path.join('run', hparams.name, 'detection_test.png'))
