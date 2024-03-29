from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from utils import specificity_score, negative_prediction_value_score, gmean_score, informedness_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class BasePulsarAI:
    def __init__(self):
        self.model_name = "PuslarAi"
        self.model_version = "1.0.0"
        self.num_runs = 3
        self.max_num_epochs = 100
        self.random_state = 42 # Random 
        self.path = './data/HTRU_2.csv'

        self.scores = {
            'accuracy': [],
            'balanced_accuracy': [],
            'recall': [],
            'specificity': [],
            'precision': [],
            'npv': [],
            'gmean': [],
            'informedness': []
        }
                

    def get_htru_2(self):
        df = pd.read_csv(self.path)
        df.columns = ['IpMean', 'IpDev', 'IpKurt','IpSkew', 'DMMean', 'DMDev', 'DMKurt', 'DMSkew', 'Class']
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=self.random_state)


        return x_train, x_test, y_train, y_test
    
    def get_torch_htru_2(self):
        x_train, x_test, y_train, y_test = self.get_htru_2()
        
        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        return x_train, x_test, y_train, y_test
    

    def append_score(self, y_test, predicted_test):
        self.scores['accuracy'].append(accuracy_score(y_test, predicted_test))
        self.scores['balanced_accuracy'].append(balanced_accuracy_score(y_test, predicted_test))
        self.scores['recall'].append(recall_score(y_test, predicted_test))
        self.scores['specificity'].append(specificity_score(y_test, predicted_test))
        self.scores['precision'].append(precision_score(y_test, predicted_test))
        self.scores['npv'].append(negative_prediction_value_score(y_test, predicted_test))
        self.scores['gmean'].append(gmean_score(y_test, predicted_test))
        self.scores['informedness'].append(informedness_score(y_test, predicted_test))

        
        
