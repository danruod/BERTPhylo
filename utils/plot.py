from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from random import sample 

def compute_roc(trues, preds, classes):
    """computes FPR, TPR, ROC AUC.

    :param trues: either ndarray of shape (n, #classes), list of class labels
                  or list of class indices
    :param preds: ndarray of shape(n, #classes)
    :param classes: list of classes
    """

    y = label_binarize(trues, classes=range(len(classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y[:, i], preds[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), preds.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for c in classes:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return namedtuple('ROC', 'fpr, tpr, roc_auc')(fpr, tpr, roc_auc)


def compute_pr(trues, preds, classes):
    """computes FPR, TPR, ROC AUC.

    :param trues: either ndarray of shape (n, #classes), list of class labels
                  or list of class indices
    :param preds: ndarray of shape(n, #classes)
    :param classes: list of classes
    """

    y = label_binarize(trues, classes=range(len(classes)))
    precision = {}
    recall = {}
    average_precision = {}

    for i, c in enumerate(classes):
        precision[c], recall[c], _ = precision_recall_curve(y[:, i], preds[:, i])
        average_precision[c] = average_precision_score(y[:, i], preds[:, i])

    # Compute micro-average ROC curve and ROC area
    precision['micro'], recall['micro'], _ = precision_recall_curve(y.ravel(), preds.ravel())
    average_precision['micro'] = average_precision_score(y, preds, average="micro")

    # First aggregate all false positive rates
    all_precision = np.unique(np.concatenate([precision[c] for c in classes]))

    # Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for c in classes:
        mean_recall+= np.interp(all_precision, precision[c], recall[c])

    # Finally average it and compute AUC
    mean_recall /= len(classes)

    precision['macro'] = all_precision
    recall['macro'] = mean_recall
    average_precision['macro'] = average_precision_score(y, preds, average="macro")

    return namedtuple('PR', 'r, p, ap')(recall, precision, average_precision)


def plot(trues, preds, classes, type='ROC', all_curves=True, fig_name=None):
    plt.figure()
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    data = {}

    if type == 'ROC':
        x_values, y_values, value = compute_roc(trues, preds, classes)
        metric_name = 'AUC'

        data['fpr'] = x_values
        data['tpr'] = y_values
        data['roc_auc'] = value

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    else:
        x_values, y_values, value = compute_pr(trues, preds, classes)
        metric_name = 'AP'

        data['recall'] = x_values
        data['precision'] = y_values
        data['average_precision'] = value

        plt.xlabel('Recall')
        plt.ylabel('Precision')

        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(f_score + 0.02, y[int(f_score * 50)] + 0.03))

    plt.plot(x_values['micro'], y_values['micro'],
             label=f"micro-average {type} curve ({metric_name} = {value['micro'] * 100:.2f}%)",
             linestyle=('--' if all_curves else '-'), linewidth=2)

    plt.plot(x_values['macro'], y_values['macro'],
             label=f"macro-average {type} curve ({metric_name}= {value['macro'] * 100:.2f})%",
             linestyle='--', linewidth=2)
    
    if (all_curves):
        sub_classes = sample(classes, 5)
        for c in sub_classes:
            plt.plot(x_values[c], y_values[c],
                     label=f'{type} curve for {c} ({metric_name} = {value[c] * 100:.2f})%')
 
    if (all_curves):
        plt.legend(loc='lower right')
    plt.tight_layout()
     
    if (fig_name):
        plt.savefig(fig_name)
        np.save(fig_name.replace('.png', '.npy'), data)
    else:
        plt.show()

    return


def visulization(labels, logits, probe, fig_path='./figs', all_curves=True):
    for i, tax_rank in enumerate(probe.tax_ranks):
        classes = list(probe.labels_maps[tax_rank].values())
        
        ood_idx = classes.index('ood')
        is_id = (labels[:, i] != ood_idx)
        classes.remove('ood')

        trues = labels[:, i][is_id].cpu().detach().numpy()
        preds = logits[i][is_id].cpu().detach().numpy()

        assert len(trues) == len(preds), f'wrong data shape: {len(trues)} != {len(preds)}'

        # plot_roc(trues, preds, classes, all_curves=all_curves, fig_name=f'{fig_path}/roc_{tax_rank}.png')

        for type in ['ROC', 'PR']:
            plot(trues, preds, classes, type, all_curves=all_curves, fig_name=f'{fig_path}/{type}_{tax_rank}.png')

    return
