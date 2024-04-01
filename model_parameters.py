from sklearn.model_selection import train_test_split
import pandas as pd
import pennylane.numpy as np
import torch
from utils import specificity_score, negative_prediction_value_score, gmean_score, informedness_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
class ModelParameters:
    def __init__(self):
        self.model_version = "1.0.0"
        self.num_runs = 3
        self.max_num_epochs = 100
        self.random_state = 42 # Random number
        self.path = './data/HTRU_2.csv'
        self.num_features = 8
        self.training_samples = 200
        self.testing_samples = 400

        self.raw_data = pd.read_csv(self.path)
        self.raw_data.columns = ['IpMean', 'IpDev', 'IpKurt','IpSkew', 'DMMean', 'DMDev', 'DMKurt', 'DMSkew', 'Class']
        self.scores = {
            'accuracy': [],
            'balanced_accuracy': [],
            'recall': [],
            'specificity': [],
            'precision': [],
            'npv': [],
            'gmean': [],
            'informedness': [],
        }

        self.times = []

    def get_htru_2(self, pi_scaler=True):
        X = self.raw_data.iloc[:, :-1].values
        y = self.raw_data.iloc[:, -1].values

        scaler = StandardScaler()
        min_max_scaler = MinMaxScaler(feature_range=(0, np.pi))
        if pi_scaler:
            X = min_max_scaler.fit_transform(X)
        else:
            X = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=self.random_state)


        return x_train, x_test, y_train, y_test
    
    def torch_convertion(self, x_train, x_test, y_train, y_test):      
        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        return x_train, x_test, y_train, y_test
    
    def generate_batches(self, x, y, batch_size):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    
    def sub_select_dataset(self, X, y, n_samples, balanced=False):
        

        # Resample to achieve the desired balance and sample size
        if balanced:
            positive_indices = y == 1
            negative_indices = y == 0
            X_pos = X[positive_indices]
            y_pos = y[positive_indices]
            X_neg = X[negative_indices]
            y_neg = y[negative_indices]
            X_pos_resampled, y_pos_resampled = resample(X_pos, y_pos, n_samples=n_samples // 2, replace=True)
            X_neg_resampled, y_neg_resampled = resample(X_neg, y_neg, n_samples=n_samples // 2, replace=True)
            # Combine and shuffle
            X_resampled = np.vstack((X_pos_resampled, X_neg_resampled))
            y_resampled = np.hstack((y_pos_resampled, y_neg_resampled))
        else:
            X_resampled, y_resampled = resample(X, y, n_samples=n_samples, replace=True)

        # Shuffle the arrays
        indices = np.arange(X_resampled.shape[0])
        np.random.shuffle(indices)

        return X_resampled[indices], y_resampled[indices]

    def append_score(self, y_test, predicted_test):
        self.scores['accuracy'].append(accuracy_score(y_test, predicted_test))
        self.scores['balanced_accuracy'].append(balanced_accuracy_score(y_test, predicted_test))
        self.scores['recall'].append(recall_score(y_test, predicted_test))
        self.scores['specificity'].append(specificity_score(y_test, predicted_test))
        self.scores['precision'].append(precision_score(y_test, predicted_test))
        self.scores['npv'].append(negative_prediction_value_score(y_test, predicted_test))
        self.scores['gmean'].append(gmean_score(y_test, predicted_test))
        self.scores['informedness'].append(informedness_score(y_test, predicted_test))

        
        
