from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
import pennylane.numpy as np

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def negative_prediction_value_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn+fn)
    return npv

def gmean_score(y_true, y_pred):
    rec = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    return np.sqrt(rec * spec)

def informedness_score(y_true, y_pred):
    rec = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    return rec + spec - 1