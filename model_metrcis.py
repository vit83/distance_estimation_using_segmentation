import torch
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def dice_coef_binary_class(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, num_of_classes):
    dice = 0

    for index in range(num_of_classes):
        y_class_true = np.equal(y_true, index)
        y_class_pred = np.equal(y_pred, index)
        dice += dice_coef_binary_class(y_class_true, y_class_pred)
    return dice / num_of_classes  # taking average


def confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat
