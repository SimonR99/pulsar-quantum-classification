import pennylane.numpy as np
from torch import FloatTensor, LongTensor
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

def sub_select_dataset(X, y, n_samples, balanced=False):
    if balanced:
        positive_indices = y == 1
        negative_indices = y == 0
        X_pos = X[positive_indices]
        y_pos = y[positive_indices]
        X_neg = X[negative_indices]
        y_neg = y[negative_indices]
        X_pos_resampled, y_pos_resampled = resample(X_pos, y_pos, n_samples=n_samples // 2, replace=True)
        X_neg_resampled, y_neg_resampled = resample(X_neg, y_neg, n_samples=n_samples // 2, replace=True)

        X_resampled = np.vstack((X_pos_resampled, X_neg_resampled))
        y_resampled = np.hstack((y_pos_resampled, y_neg_resampled))
    else:
        X_resampled, y_resampled = resample(X, y, n_samples=n_samples, replace=True)

    indices = np.arange(X_resampled.shape[0])
    np.random.shuffle(indices)

    return X_resampled[indices], y_resampled[indices]


def torch_convertion(x_train, x_test, y_train, y_test):      
    x_train = FloatTensor(x_train)
    x_test = FloatTensor(x_test)
    y_train = LongTensor(y_train)
    y_test = LongTensor(y_test)
    return x_train, x_test, y_train, y_test


def generate_batches(x, y, batch_size):
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
