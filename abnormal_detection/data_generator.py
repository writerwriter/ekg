import numpy as np
import keras
from sklearn.model_selection import train_test_split

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.data_utils import patient_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_PATIENT_X = 852

class DataGenerator:
    def __init__(self):
        self.X, self.y = None, None
        self.patient_X = np.load(os.path.join(DATA_DIR, 'patient_X.npy')) # (NUM_PATIENT_X, 10, 10000)
        self.normal_X = np.load(os.path.join(DATA_DIR, 'normal_X.npy')) # (?, 10, 10000)
        self.preprocessing()

    def preprocessing(self):
        # combine normal and patient
        self.X = np.append(self.patient_X, self.normal_X, axis=0) # (?, 10, 10000)
        self.y = np.array([1]*self.patient_X.shape[0] + [0]*self.normal_X.shape[0])

        # change dimension
        # ?, n_channels, n_points -> ?, n_points, n_channels
        self.X = np.swapaxes(self.X, 1, 2)

        print('baseline:', self.y.sum() / self.y.shape[0])
        self.y = keras.utils.to_categorical(self.y, num_classes=2) # to one-hot

    def get(self):
        train_set, valid_set, test_set = self.split(self.X, self.y)

        # do normalize using means and stds from training data
        train_set[0], self.means_and_stds = DataGenerator.normalize(train_set[0])
        valid_set[0], _ = DataGenerator.normalize(valid_set[0], self.means_and_stds)
        test_set[0], _ = DataGenerator.normalize(test_set[0], self.means_and_stds)

        return train_set, valid_set, test_set

    @staticmethod
    def normalize(X, means_and_stds=None):
        if means_and_stds is None:
            means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
            stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
        else:
            means = means_and_stds[0]
            stds = means_and_stds[1]

        normalized_X = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[-1]):
            normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
            normalized_X[..., i] = normalized_X[..., i] / stds[i]
        return normalized_X, (means, stds)

    @staticmethod
    def split(X, y, rs=42):
        # do patient split
        patient_training_set, patient_valid_set, patient_test_set  = patient_split(X[:NUM_PATIENT_X, ...], y[:NUM_PATIENT_X, ...], rs)

        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(X[NUM_PATIENT_X:, ...], y[NUM_PATIENT_X:, ...], test_size=0.3, random_state=rs)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=rs)

        # combine
        X_train = np.append(X_train, patient_training_set[0], axis=0)
        y_train = np.append(y_train, patient_training_set[1], axis=0)

        X_valid = np.append(X_valid, patient_valid_set[0], axis=0)
        y_valid = np.append(y_valid, patient_valid_set[1], axis=0)

        X_test = np.append(X_test, patient_test_set[0], axis=0)
        y_test = np.append(y_test, patient_test_set[1], axis=0)

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
