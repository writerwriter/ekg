import numpy as np
import keras
import sklearn
from sklearn.model_selection import train_test_split

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.data_utils import patient_split, DataAugmenter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class DataGenerator:
    def __init__(self, remove_dirty=0, ekg_scaling_prob=0.5, hs_scaling_prob=0.5, time_stretch_prob=0.):
        self.X, self.y = None, None
        if remove_dirty > 0:
            self.patient_X = np.load(os.path.join(DATA_DIR, 'cleaned_{:d}_patient_X.npy'.format(remove_dirty)))
            self.patient_id = np.load(os.path.join(DATA_DIR, 'cleaned_{:d}_patient_id.npy'.format(remove_dirty)))
        else:
            self.patient_X = np.load(os.path.join(DATA_DIR, 'patient_X.npy')) # (?, 10, 10000)
            self.patient_id = np.load(os.path.join(DATA_DIR, 'patient_id.npy'))

        self.normal_X = np.load(os.path.join(DATA_DIR, 'normal_X.npy')) # (?, 10, 10000)
        self.preprocessing()
        self.augmenter = DataAugmenter(indices_channel_ekg=[0, 1, 2, 3, 4, 5, 6, 7], indices_channel_hs=[8, 9],
                                        ekg_scaling_prob=ekg_scaling_prob, hs_scaling_prob=hs_scaling_prob,
                                        time_stretch_prob=time_stretch_prob)

    def preprocessing(self):
        # combine normal and patient
        self.X = np.append(self.patient_X, self.normal_X, axis=0) # (?, 10, 10000)
        self.y = np.array([1]*self.patient_X.shape[0] + [0]*self.normal_X.shape[0])

        # change dimension
        # ?, n_channels, n_points -> ?, n_points, n_channels
        self.X = np.swapaxes(self.X, 1, 2) # (?, 10000, 10)

        print('baseline:', self.y.sum() / self.y.shape[0])
        self.y = keras.utils.to_categorical(self.y, num_classes=2) # to one-hot

    def get(self):
        train_set, valid_set, test_set = self.split()

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

    def split(self, rs=42):
        number_patient = self.patient_X.shape[0]

        # do patient split
        patient_training_set, patient_valid_set, patient_test_set  = patient_split(self.X[:number_patient], self.y[:number_patient], self.patient_id, rs)

        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(self.X[number_patient: ], self.y[number_patient: ], test_size=0.3, random_state=rs)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=rs)

        # combine
        X_train = np.append(X_train, patient_training_set[0], axis=0)
        y_train = np.append(y_train, patient_training_set[1], axis=0)

        X_valid = np.append(X_valid, patient_valid_set[0], axis=0)
        y_valid = np.append(y_valid, patient_valid_set[1], axis=0)

        X_test = np.append(X_test, patient_test_set[0], axis=0)
        y_test = np.append(y_test, patient_test_set[1], axis=0)

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]

    def augment_batch(self, X, y):
        for i, Xi in enumerate(X):
            X[i] = self.augmenter.augment(Xi)
        return X, y

    def batch_generator(self, X, y, batch_size, shuffle=True, data_augmentation=True):
        while True:
            if shuffle:
                X, y = sklearn.utils.shuffle(X, y)

            for index_batch in range(int(np.ceil(X.shape[0] / batch_size))):
                index_start = int(index_batch * batch_size)
                index_end = min(X.shape[0], index_start + batch_size)
                m = np.s_[index_start: index_end]
                if data_augmentation:
                    yield self.augment_batch(X[m], y[m])
                else:
                    yield X[m], y[m]
