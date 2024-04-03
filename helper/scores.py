
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
import pennylane.numpy as np

class Metrics:
    def __init__(self):
        self.num_digits = 3
        self.accuracy = []
        self.balanced_accuracy = []
        self.recall = []
        self.specificity = []
        self.precision = []
        self.npv = []
        self.gmean = []
        self.informedness = []
        self.training_duration = []
        self.testing_duration = []


    def append_score(self, y_test, predicted_test, training_duration, testing_duration):
        self.accuracy.append(accuracy_score(y_test, predicted_test))
        self.balanced_accuracy.append(balanced_accuracy_score(y_test, predicted_test))
        self.recall.append(recall_score(y_test, predicted_test))
        self.specificity.append(specificity_score(y_test, predicted_test))
        self.precision.append(precision_score(y_test, predicted_test))
        self.npv.append(negative_prediction_value_score(y_test, predicted_test))
        self.gmean.append(gmean_score(y_test, predicted_test))
        self.informedness.append(informedness_score(y_test, predicted_test))
        self.training_duration.append(training_duration)
        self.testing_duration.append(testing_duration)
    
    def display(self):
        format_specifier = f".{self.num_digits}f"
        print("Pour " + str(len(self.accuracy)) + " runs:")
        print(f"Accuracy: {np.mean(self.accuracy):{format_specifier}} ± {np.std(self.accuracy):{format_specifier}}")
        print(f"Balanced Accuracy: {np.mean(self.balanced_accuracy):{format_specifier}} ± {np.std(self.balanced_accuracy):{format_specifier}}")
        print(f"Recall: {np.mean(self.recall):{format_specifier}} ± {np.std(self.recall):{format_specifier}}")
        print(f"Specificity: {np.mean(self.specificity):{format_specifier}} ± {np.std(self.specificity):{format_specifier}}")
        print(f"Precision: {np.mean(self.precision):{format_specifier}} ± {np.std(self.precision):{format_specifier}}")
        print(f"NPV: {np.mean(self.npv):{format_specifier}} ± {np.std(self.npv):{format_specifier}}")
        print(f"G-Mean: {np.mean(self.gmean):{format_specifier}} ± {np.std(self.gmean):{format_specifier}}")
        print(f"Informedness: {np.mean(self.informedness):{format_specifier}} ± {np.std(self.informedness):{format_specifier}}")
        print(f"Training Duration: {np.mean(self.training_duration):{format_specifier}} ± {np.std(self.training_duration):{format_specifier}}")
        print(f"Testing Duration: {np.mean(self.testing_duration):{format_specifier}} ± {np.std(self.testing_duration):{format_specifier}}")


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