import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          classes = None,
                          title=None,
                          figname='confusion_matrix.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = np.ceil(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100)

    figsize = 3
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    fig.savefig(figname)
    return ax, cm


def plot_roc(y_test, y_pred_prob_vec, n_classes, myplt=None, color='darkorange', modelname=''):
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_vec = []
    for i in y_test.tolist():
        if i == 0:
            y_test_vec.append([1, 0])
        else:
            y_test_vec.append([0, 1])
    y_test_vec = np.array(y_test_vec)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_vec[:, i], y_pred_prob_vec[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_vec.ravel(), y_pred_prob_vec.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    myplt.figure()
    lw = 2
    myplt.plot(fpr[1], tpr[1], color=color,
             lw=lw, label='%s ROC curve (area = %0.2f)' % (modelname, roc_auc[1]))
    myplt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    myplt.xlim([0.0, 1.0])
    myplt.ylim([0.0, 1.05])
    myplt.xlabel('False Positive Rate')
    myplt.ylabel('True Positive Rate')
    myplt.title('Receiver operating characteristic example')
    myplt.legend(loc="lower right")

    #myplt.savefig(figname)

