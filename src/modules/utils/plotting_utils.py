from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def plot_confusion_matrix(y_true, y_pred, lst_labels, dest_path=None):
    """
    Plots confusion matrix with seaborn heatmap.

    :param y_true: Array of numeric gt labels
    :param y_pred: Array of numeric predicted labels
    :param lst_labels: List of string label names
    :return:
    """
    labels_str = [lst_labels[x] for x in y_true]
    predicted_str = [lst_labels[x] for x in y_pred]

    arr_cm = confusion_matrix(labels_str, predicted_str, labels=lst_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    ax_cm = sn.heatmap(arr_cm, annot=True, xticklabels=lst_labels, yticklabels=lst_labels, fmt='d')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')

    if dest_path is not None:
        fig_cm.savefig(dest_path)

    return fig_cm


def plot_roc_curve(y_true, y_pred, dest_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if dest_path is not None:
        fig_roc.savefig(dest_path)

    return fig_roc


def plot_pr(precision, recall, target_labels, y_label='Scores', lst_metric_names=('Precision', 'Recall'), dest_path=None):
    ind = np.arange(len(target_labels))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig_pr = plt.figure(figsize=(11, 11))
    ax = fig_pr.add_subplot(111)
    rects1 = ax.bar(ind, precision, width, color='royalblue')
    rects2 = ax.bar(ind + width, recall, width, color='seagreen')
    ax.set_ylabel(y_label)
    ax.set_title('Scores by class')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(target_labels)
    ax.legend((rects1[0], rects2[0]), lst_metric_names)
    plt.xticks(rotation=90)

    if dest_path is not None:
        fig_pr.savefig(dest_path)

    return fig_pr
