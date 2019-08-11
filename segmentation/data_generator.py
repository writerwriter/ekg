import numpy as np
from sklearn.model_selection import train_test_split

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class DataGenerator:
    def __init__(self, config):
        self.patient_X = np.load(os.path.join(DATA_DIR, 'seg_X.npy')) # (90, 8, 10000)
        self.patient_y= np.load(os.path.join(DATA_DIR, 'seg_y.npy')) # (90, 6, 10000)

        self.means_and_stds = None
        self.config = config

        self.X, self.y = None, None
        self.preprocessing()

    def preprocessing(self):
        self.X = np.swapaxes(self.patient_X, 1, 2) # (90, 10000, 8)
        self.y = np.swapaxes(self.patient_y, 1, 2) # (90, 10000, 6) # pqrst

        if self.config.target == '24hr':
            # downsample the signals to 500hz, while the original ones are 1000Hz
            self.X = self.X[:, ::2, 0:1] # and only use the first lead signals
            self.y = self.y[:, ::2, :] + self.y[:, 1::2, :]
        else: # bigexam
            pass

        if self.config.seg_setting == 'split':
            print('ERROR: seg_setting {} not supported!'.format(self.config.seg_setting))

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

    def get(self):
        train_set, valid_set, test_set = self.split()

        # do normalize using means and stds from training data
        train_set[0], self.means_and_stds = DataGenerator.normalize(train_set[0])
        valid_set[0], _ = DataGenerator.normalize(valid_set[0], self.means_and_stds)
        test_set[0], _ = DataGenerator.normalize(test_set[0], self.means_and_stds)

        return train_set, valid_set, test_set

    def split(self, rs=42):
        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=rs)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=rs)

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
