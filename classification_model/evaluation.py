import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, \
                            precision_score, recall_score, f1_score


def generate_report(y_true: list, y_pred: list):
    report = np.round(pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)), 2).T
    return report


def confusion_matrix(y_true: list, y_pred: list, normalize: bool = True):
    if normalize:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred,
                                     rownames=['Observed'], colnames=['Predicted'], normalize='index'), 2)
    else:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred, rownames=['Observed'], colnames=['Predicted']), 2)
    return table


def metrics_summary(y_true: list, y_pred: list):
    metrics = {
        'accuracy': np.round(accuracy_score(y_true=y_true, y_pred=y_pred), 2),
        'precision': np.round(precision_score(y_true=y_true, y_pred=y_pred, average='weighted'), 2),
        'recall': np.round(recall_score(y_true=y_true, y_pred=y_pred, average='weighted'), 2),
        'f1': np.round(f1_score(y_true=y_true, y_pred=y_pred, average='weighted'), 2),

    }
    print(f'The accuracy is: {metrics["accuracy"]}')
    print(f'The precision is: {metrics["precision"]}')
    print(f'The recall is: {metrics["recall"]}')
    print(f'The F1 score is: {metrics["f1"]}')


def print_metrics_summary(y_test: list, y_pred: list):
    for i, col in enumerate(y_test):
        print(f'The metrics for {col} is: \n')
        print(metrics_summary(y_test[col], y_pred[:, i]))


def print_report(y_test: list, y_pred: list):
    for i, col in enumerate(y_test):
        print(f'The prediction for {col} is: \n')
        print(generate_report(y_test[col], y_pred[:, i]))


def print_cm(y_test: list, y_pred: list):
    for i, col in enumerate(y_test):
        print(f'The confusion matrix for {col} is: \n')
        print(confusion_matrix(y_test[col], y_pred[:, i]))
