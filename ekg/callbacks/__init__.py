import wandb
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from lifelines.utils import concordance_index
import gc

class LogBest(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', records=['val_loss', 'loss', 'val_acc', 'acc']):
        self.monitor = monitor
        self.records = records

        setattr(self, 'best_' + self.monitor, np.inf if 'loss' in self.monitor else 0)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if getattr(self, 'best_' + self.monitor) > logs.get(self.monitor): # update
            setattr(self, 'best_' + self.monitor, logs.get(self.monitor))

            log_dict = dict()
            for rs in self.records:
                log_dict['best_' + rs] = logs.get(rs)
            log_dict['best_epoch'] = epoch

            wandb.log(log_dict, commit=False)

class ConcordanceIndex(keras.callbacks.Callback):
    def __init__(self, train_set, valid_set, event_names, prediction_model, reverse=True):
        super().__init__()
        self.train_set, self.valid_set = train_set, valid_set
        self.event_names = event_names
        self.prediction_model = prediction_model
        self.reverse = reverse

    def on_epoch_end(self, epoch, logs={}):
        X_train, y_train = self.train_set[0], self.train_set[1]
        X_valid, y_valid = self.valid_set[0], self.valid_set[1]

        pred_train = self.prediction_model.predict(X_train) # (?, n_events)
        pred_valid = self.prediction_model.predict(X_valid)

        gc.collect() # try to fix memory leak

        if self.reverse:
            pred_train = pred_train * -1
            pred_valid = pred_valid * -1

        for i in range(len(self.event_names)):
            cs_train, st_train = y_train[:, i, 0], y_train[:, i, 1]
            cs_valid, st_valid = y_valid[:, i, 0], y_valid[:, i, 1]

            try:
                train_cindex = concordance_index(st_train, pred_train[:, i], cs_train)
            except ZeroDivisionError:
                train_cindex = np.nan
            
            try:
                valid_cindex = concordance_index(st_valid, pred_valid[:, i], cs_valid)
            except ZeroDivisionError:
                valid_cindex = np.nan

            print('Concordance index of {} training set: {:.4f}'.format(self.event_names[i], train_cindex))
            print('Concordance index of {} validation set: {:.4f}'.format(self.event_names[i], valid_cindex))

            # append cindex to logs
            logs['{}_cindex'.format(self.event_names[i])] = train_cindex
            logs['val_{}_cindex'.format(self.event_names[i])] = valid_cindex

class LossVariableChecker(keras.callbacks.Callback):
    def __init__(self, event_names):
        self.event_names = event_names

    def on_epoch_end(self, epoch, logs={}):
        if hasattr(self.model.layers[-1], 'log_sigma'):
            for log_sigma, event in zip(self.model.layers[-1].log_sigma, self.event_names):
                sigma = np.exp(K.get_value(log_sigma))[0]

                print('{} sigma: {:.4f}'.format(event, sigma))
                logs['{}_sigma'.format(event)] = sigma

class GarbageCollector(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        gc.collect() # try to fix memory leak