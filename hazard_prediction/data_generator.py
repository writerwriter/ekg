import numpy as np
import keras
import copy
import sklearn
from sklearn.model_selection import train_test_split

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.data_utils import patient_split, DataAugmenter
from ekg.utils.data_utils import load_survival_data

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class DataGenerator:
    def __init__(self, remove_dirty=0, event_names=['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality'],
                        ekg_scaling_prob=0., hs_scaling_prob=0., time_stretch_prob=0.):
        self.remove_dirty = remove_dirty
        self.X, self.y = None, None
        if remove_dirty > 0:
            self.patient_X = np.load(os.path.join(DATA_DIR, 'cleaned_{:d}_patient_X.npy'.format(remove_dirty)))
            self.patient_id = np.load(os.path.join(DATA_DIR, 'cleaned_{:d}_patient_id.npy'.format(remove_dirty)))
        else:
            self.patient_X = np.load(os.path.join(DATA_DIR, 'patient_X.npy')) # (?, 10, 10000)
            self.patient_id = np.load(os.path.join(DATA_DIR, 'patient_id.npy'))

        self.patient_y = self.load_patient_target(event_names)  # (?, n_events, 2)

        self.normal_X = np.load(os.path.join(DATA_DIR, 'normal_X.npy')) # (?, 10, 10000)
        self.normal_y = self.load_normal_target(event_names)    # (?, n_events, 2)

        self.preprocessing()
        self.augmenter = DataAugmenter(indices_channel_ekg=[0, 1, 2, 3, 4, 5, 6, 7], indices_channel_hs=[8, 9],
                                        ekg_scaling_prob=ekg_scaling_prob, hs_scaling_prob=hs_scaling_prob,
                                        time_stretch_prob=time_stretch_prob)

    def load_patient_target(self, event_names):
        patient_y = np.zeros((self.patient_X.shape[0], len(event_names), 2)) # (?, n_events, 2)
        for i, en in enumerate(event_names):
            cs, st = load_survival_data(en, self.remove_dirty)
            patient_y[:, i, 0] = cs
            patient_y[:, i, 1] = st
        return patient_y

    def load_normal_target(self, event_names): # NOTE: must be done after generating patient_y
        normal_y = np.zeros((self.normal_X.shape[0], len(event_names), 2)) # (?, n_events, 2)
        # normal_y[:, i, 0] = cs = 0 -> survived
        for i, en in enumerate(event_names):
            normal_y[:, i, 1] = self.patient_y[:, i, 1].max() # maximum survival dur of the event
        return normal_y

    def clean_patient(self):
        remove_mask = np.zeros((self.patient_X.shape[0], ), dtype=bool) # all False
        for i in range(self.patient_y.shape[1]): # number of events
            remove_mask = np.logical_or(remove_mask, self.patient_y[:, i, 0] == -1) # remove rows with cs == -1

        keep_mask = ~remove_mask
        self.patient_X = self.patient_X[keep_mask]
        self.patient_y = self.patient_y[keep_mask]
        self.patient_id = self.patient_id[keep_mask]

    def preprocessing(self):
        self.clean_patient()

        # combine normal and patient
        self.X = np.append(self.patient_X, self.normal_X, axis=0) # (?, 10, 10000)
        self.y = np.append(self.patient_y, self.normal_y, axis=0) # (?, n_events, 2)

        # change dimension
        # ?, n_channels, n_points -> ?, n_points, n_channels
        self.X = np.swapaxes(self.X, 1, 2) # (?, 10000, 10)

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

    @staticmethod
    def shuffle(X, y, batch_size):
        def sort_batch(X, cs, st):
            sorting_indices = np.argsort(st)[::-1]
            X = X[sorting_indices]
            cs = cs[sorting_indices]
            st = st[sorting_indices]
            return X, cs, st

        # copy may not be needed, but is done anyway
        X, y = copy.deepcopy(X), copy.deepcopy(y)
        cs, st = y[:, 0], y[:, 1]

        # shuffle X and y
        shuffled_indices = np.random.choice(list(range(X.shape[0])), size=X.shape[0], replace=False)
        X, cs, st = X[shuffled_indices], cs[shuffled_indices], st[shuffled_indices]

        for index_batch in range(int(np.ceil(X.shape[0] / batch_size))):
            index_start = int(index_batch * batch_size)
            index_end = min(X.shape[0], index_start + batch_size)
            m = np.s_[index_start: index_end]
            X[m], cs[m], st[m] = sort_batch(X[m], cs[m], st[m])

        y = np.array([cs, st]).T
        return X, y

    def batch_generator(self, X, y, batch_size, shuffle=True, data_augmentation=False):
        while True:
            if shuffle:
                X, y = self.shuffle(X, y, batch_size)

            for index_batch in range(int(np.ceil(X.shape[0] / batch_size))):
                index_start = int(index_batch * batch_size)
                index_end = min(X.shape[0], index_start + batch_size)
                m = np.s_[index_start: index_end]
                if data_augmentation:
                    yield self.augment_batch(X[m], y[m])
                else:
                    yield X[m], y[m]
