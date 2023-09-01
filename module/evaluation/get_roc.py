'''
Use to draw ROC curve
'''

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def get_roc_ret(score, label, pos_label=1,
                tolerances=[10 ** -4, 5e-4, 10 ** -3, 2e-3, 4e-3, 6e-3, 10 ** -2, 5e-2, 10 ** -1]):

    fpr, tpr, thresholds = roc_curve(label, score, pos_label=pos_label)
    roc_info = []
    info_dict = {}

    # 10**-6,10**-5, 10**-4, 10**-3, 10**-2, 10**-1
    for tolerance in tolerances:
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))

        #save roc info log
        temp_info = 'fpr\t{}\tacc\t{:.5f}\tthreshold\t{:.6f}'.format(tolerance, max_acc, threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = max_acc
    return roc_info, info_dict

