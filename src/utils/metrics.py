import numpy as np

def f1score(TP, TN, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)

def recall(TP, TN, FP, FN):
    return TP / (TP + FN)

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

def precision(TP, TN, FP, FN):
    return TP / (TP + FP)

def specificity(TP, TN, FP, FN):
    return TN / (TN + FP)

def matthews_correlation_coefficient(TP, TN, FP, FN):
    return (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * ( TN + FP) * (TN + FN))