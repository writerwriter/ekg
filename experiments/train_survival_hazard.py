import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use 0-th gpu

import numpy as np
import better_exceptions; better_exceptions.hook()
import keras
import sklearn.metrics
import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from models import get_survival_hazard_model
import utils
from utils import patient_split

from datetime import datetime
import pickle
import copy
import lifelines
from lifelines.utils import concordance_index

import sys

class DataGenerator:
    def __init__(self):
        self.patient_X = np.load('./patient_X.npy') # (852, 10, 10000)
        self.normal_X = np.load('./normal_X.npy') # (103, 10, 10000)

        self.patient_censoring_stats, self.patient_survival_times = utils.load_survival_data()
        self.preprocessing()

    def preprocessing(self):
        # remove ignoring data
        keep_indices = (self.patient_censoring_stats != -1)
        self.patient_X = self.patient_X[keep_indices]
        self.patient_censoring_stats = self.patient_censoring_stats[keep_indices]
        self.patient_survival_times = self.patient_survival_times[keep_indices]

        # combine normal and patient
        self.X = np.append(self.patient_X, self.normal_X, axis=0) # (?, 10, 10000)

        # generate normal survival data
        normal_censoring_stats = np.zeros((self.normal_X.shape[0], )) # survived
        nromal_survival_times = np.ones((self.normal_X.shape[0], )) * (self.patient_survival_times.max() + 1)

        # combine cencsoring stats and survival times
        self.cencsoring_stats = np.append(self.patient_censoring_stats, normal_censoring_stats, axis=0)
        self.survival_times = np.append(self.patient_survival_times, nromal_survival_times, axis=0)

        # change dimension
        # ?, n_channels, n_points -> ?, n_points, n_channels
        self.X = np.swapaxes(self.X, 1, 2)

        self.y = (self.cencsoring_stats, self.survival_times)

    def X_shape(self):
        return self.X.shape[1:]

    @staticmethod
    def print_baseline(y):
        print('baseline:', y.sum() / y.shape[0])

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

    def data(self):
        return self.X, self.y

    @staticmethod
    def split(X, y, rs=42):
        cs, st = y

        # do random split
        X_train, X_test, cs_train, cs_test, st_train, st_test = train_test_split(X, cs, st, test_size=0.3, random_state=42)
        X_train, X_valid, cs_train, cs_valid, st_train, st_valid = train_test_split(X_train, cs_train, st_train, test_size=0.3, random_state=42)

        y_train = [cs_train, st_train]
        y_valid = [cs_valid, st_valid]
        y_test = [cs_test, st_test]

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]

    @staticmethod
    def shuffle(X, y, batch_size):
        def sort_batch(X, cs, st):
            sorting_indices = np.argsort(st)[::-1]
            X = X[sorting_indices]
            cs = cs[sorting_indices]
            st = st[sorting_indices]
            return X, cs, st

        # copy may not be needed, but is done anyway
        X, (cs, st) = copy.deepcopy(X), copy.deepcopy(y)

        # shuffle X and y
        shuffled_indices = np.random.choice(list(range(X.shape[0])), size=X.shape[0], replace=False)
        X = X[shuffled_indices]
        cs = cs[shuffled_indices]
        st = st[shuffled_indices]

        for index_batch in range(int(np.ceil(X.shape[0] / batch_size))):
            index_start = int(index_batch * batch_size)
            index_end = min(X.shape[0], index_start + batch_size)
            m = np.s_[index_start: index_end]
            X[m], cs[m], st[m] = sort_batch(X[m], cs[m], st[m])

        return X, cs, st

    @staticmethod
    def batch_generator(X, y, batch_size, shuffle=True):
        while True:
            if shuffle:
                this_X, cs, st = DataGenerator.shuffle(X, y, batch_size)
            else:
                this_X, (cs, st) = X, y

            for index_batch in range(int(np.ceil(X.shape[0] / batch_size))):
                index_start = int(index_batch * batch_size)
                index_end = min(X.shape[0], index_start + batch_size)
                m = np.s_[index_start: index_end]
                yield this_X[m], cs[m]

class ConcordanceIndex(Callback):
    def __init__(self, train_set, valid_set):
        super(ConcordanceIndex, self).__init__()
        self.train_set = train_set
        self.valid_set = valid_set

    def on_epoch_end(self, epoch, logs={}):
        X_train, (cs_train, st_train) = self.train_set
        X_valid, (cs_valid, st_valid) = self.valid_set
        train_cindex = concordance_index(st_train, -self.model.predict(X_train), cs_train)
        valid_cindex = concordance_index(st_valid, -self.model.predict(X_valid), cs_valid)
        print('Concordance index of training set: {:.4f}'.format(train_cindex))
        print('Concordance index of validation set: {:.4f}'.format(valid_cindex))

def train():
    g = DataGenerator()
    X, y = g.data()
    train_set, valid_set, test_set = g.split(X, y)

    model_checkpoints_dirname = 'survival_hazard_model_checkpoints/'+datetime.now().strftime('%Y%m%d_%H%M_%S')
    tensorboard_log_dirname = model_checkpoints_dirname + '/logs'
    os.makedirs(model_checkpoints_dirname)
    os.makedirs(tensorboard_log_dirname)

    # do normalize using means and stds from training data
    train_set[0], means_and_stds = DataGenerator.normalize(train_set[0])
    valid_set[0], _ = DataGenerator.normalize(valid_set[0], means_and_stds)
    test_set[0], _ = DataGenerator.normalize(test_set[0], means_and_stds)

    # save means and stds
    with open(model_checkpoints_dirname + '/means_and_stds.pl', 'wb') as f:
        pickle.dump(means_and_stds, f)

    model = get_survival_hazard_model()
    model.summary()

    # shuffle and sort validation set first
    valid_set[0], valid_set[1][0], valid_set[1][1] = g.shuffle(valid_set[0], valid_set[1], batch_size=64)
    callbacks = [
        # EarlyStopping(patience=5),
        ConcordanceIndex(train_set, valid_set),
        ModelCheckpoint(model_checkpoints_dirname + '/{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        TensorBoard(log_dir=tensorboard_log_dirname)
    ]

    model.fit_generator(g.batch_generator(train_set[0], train_set[1], batch_size=64),
                 steps_per_epoch=int(np.ceil(train_set[0].shape[0] / 64)),
                 epochs=500, 
                 validation_data=(valid_set[0], valid_set[1][0]), 
                 callbacks=callbacks, shuffle=False)

    X_test, (y_test, st_test) = test_set[0], test_set[1]
    print('Concordance index of validation set:', concordance_index(st_test, -model.predict(X_test), y_test))

def testing():
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    # testing()
    train()
