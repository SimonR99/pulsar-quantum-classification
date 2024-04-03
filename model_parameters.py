from sklearn.model_selection import train_test_split
import pandas as pd
import pennylane.numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ModelParameters:
    def __init__(self):
        self.model_version = "1.0.0"
        self.num_runs = 6
        self.max_num_epochs = 150
        self.random_state = 42 # Random number
        self.path = './data/HTRU_2.csv'
        self.num_features = 8
        self.training_samples = 200
        self.testing_samples = 400

        self.raw_data = pd.read_csv(self.path)
        self.raw_data.columns = ['IpMean', 'IpDev', 'IpKurt','IpSkew', 'DMMean', 'DMDev', 'DMKurt', 'DMSkew', 'Class']
     
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
        
