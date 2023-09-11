import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


def print_anomaly_stats(frame):
    """
    Print results of anomaly detection as confusion matrix
    """
    # convert boolean classification to anomaly class classification
    # NOTE: Probably not the best solution
    anomalies = frame.anomalies.values != 0
    anomalies_hat = frame.anomalies_hat.values
    prediction = np.zeros(frame.anomalies.shape, dtype=int)
    prediction[anomalies & anomalies_hat] = frame.anomalies[anomalies & anomalies_hat]
    prediction[~anomalies & anomalies_hat] = -1

    # print results and debug information
    anomalies = frame.anomalies.values
    anomalies_hat = prediction
    conf = confusion_matrix(frame.anomalies, anomalies_hat)
    labels = np.sort(np.unique(np.append(anomalies, anomalies_hat)))
    header = '   gt\pred  ' + '  '.join([f'Class {label:2}' for label in labels])
    print(header)
    for i, data in enumerate(conf):
        line = f'Class {labels[i]:2}     ' + '   '.join([f'{x:7}' for x in data])
        print(line)
    print(classification_report(frame.anomalies, anomalies_hat))
