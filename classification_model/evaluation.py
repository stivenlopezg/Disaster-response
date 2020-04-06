import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, \
                            precision_score, recall_score, f1_score


def generate_report(y_true: list, y_pred: list):
    """
    Generate a DataFrame with the main classification metrics
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: pd.DataFrame
    """
    report = np.round(pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)), 2).T
    return report


def confusion_matrix(y_true: list, y_pred: list, normalize: bool = True):
    """
    Generates a confusion matrix
    :param y_true: Observed value
    :param y_pred: Predicted value
    :param normalize: Boolean on normalize for each row
    :return: pd.DataFrame
    """
    if normalize:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred,
                                     rownames=['Observed'], colnames=['Predicted'], normalize='index'), 2)
    else:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred, rownames=['Observed'], colnames=['Predicted']), 2)
    return table


def metrics_summary(y_true: list, y_pred: list):
    """
    Text summary with the metrics accuracy, precision, recall and F1
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: string
    """
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
    """
    Print a summary of the metrics for each disaster category
    :param y_test: Observed value
    :param y_pred: Predcited value
    :return: string
    """
    for i, col in enumerate(y_test):
        print(f'The metrics for {col} are: \n')
        print(metrics_summary(y_test[col], y_pred[:, i]))


def print_report(y_test: list, y_pred: list):
    """
    Print the classification report for each disaster category
    :param y_test: Observed value
    :param y_pred: Predicted value
    :return: string
    """
    for i, col in enumerate(y_test):
        print(f'The prediction for {col} are: \n')
        print(generate_report(y_test[col], y_pred[:, i]))


def print_cm(y_test: list, y_pred: list):
    """
    Print the confusion matrix for each disaster category
    :param y_test: Observed value
    :param y_pred: Predicted value
    :return: string
    """
    for i, col in enumerate(y_test):
        print(f'The confusion matrix for {col} are: \n')
        print(confusion_matrix(y_test[col], y_pred[:, i]))
