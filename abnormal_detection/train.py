#!/usr/bin/env python3
import numpy as np
import keras
import sklearn.metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle
import better_exceptions; better_exceptions.hook()

# allow_growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import keras
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D, ReLU, Bidirectional, Maximum
from keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate, Flatten, Dropout, LSTM, Reshape, SeparableConv1D
from keras.utils import conv_utils

from ekg.layers import LeftCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.layers.non_local import non_local_block
from ekg.utils.data_utils import patient_split
from ekg.utils.eval_utils import print_cm

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import wandb
from wandb.keras import WandbCallback
wandb.init(project='ekg-abnormal_detection', entity='toosyou')

def set_wandb_config(params, overwrite=False):
    for key, value in params.items():
        if key not in wandb.config._items or overwrite:
            wandb.config.update({key: value})

set_wandb_config({
    'sincconv_filter_length': 31,
    'sincconv_nfilters': 16,
    'ekg_branch_nlayers': 2,
    'ekg_kernel_length': 7,

    'hs_branch_nlayers': 2,
    'hs_kernel_length': 7,

    'final_nlayers': 7,
    'final_kernel_length': 7,
    'final_nonlocal_nlayers': 2
})

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_model():
    def heart_sound_branch(hs):
        sincconv_filter_length = wandb.config.sincconv_filter_length - (wandb.config.sincconv_filter_length+1) % 2
        hs = SincConv1D(wandb.config.sincconv_nfilters, sincconv_filter_length, 1000)(hs)
        hs = BatchNormalization()(hs)

        for _ in wandb.config.hs_branch_nlayers:
            hs = Conv1D(8, wandb.config.hs_kernel_length, activation='relu', padding='same')(hs)
            hs = BatchNormalization()(hs)
            hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)
        return hs

    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    # ekg branch
    ekg = ekg_input
    for _ in range(wandb.config.ekg_branch_nlayers):
        ekg = Conv1D(8, wandb.config.ekg_kernel_length, activation='relu', padding='same')(ekg_input)
        ekg = BatchNormalization()(ekg)
        ekg = MaxPooling1D(3, padding='same')(ekg)

    # heart sound branch
    hs_outputs = list()
    hs = Lambda(lambda x: K.expand_dims(x[:, :, 0], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 1], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Add()(hs_outputs) # (?, 1625, 128)
    ekg = LeftCropLike()([ekg, hs])
    output = Concatenate(axis=-1)([hs, ekg])

    # final layers
    for i in range(wandb.config.final_nlayers):
        output = Conv1D(8, wandb.config.final_kernel_length, activation='relu', padding='same')(output)
        output = BatchNormalization()(output)

        if i >= wandb.config.final_nlayers - wandb.config.final_nonlocal_nlayers: # the final 'final_nonlocal_nlayers' layers
            output = output = non_local_block(output, compression=2, mode='embedded')

        if i != wandb.config.final_nlayers-1: # not the final output
            output = MaxPooling1D(2, padding='same')(output) # 829

    output = GlobalAveragePooling1D()(output)
    output = Dense(2, activation='softmax')(output)

    model = Model(total_input, output)
    model.compile(Adam(amsgrad=True), 'binary_crossentropy', metrics=['acc'])
    return model

class DataGenerator:
    def __init__(self):
        self.patient_X = np.load(os.path.join(DATA_DIR, 'patient_X.npy')) # (852, 10, 10000)
        self.normal_X = np.load(os.path.join(DATA_DIR, 'normal_X.npy')) # (103, 10, 10000)
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

    def X_shape(self):
        return self.X.shape[1:]

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
        # do patient split
        patient_training_set, patient_valid_set, patient_test_set  = patient_split(X[:852, ...], y[:852, ...])

        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(X[852:, ...], y[852:, ...], test_size=0.3, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

        # combine
        X_train = np.append(X_train, patient_training_set[0], axis=0)
        y_train = np.append(y_train, patient_training_set[1], axis=0)

        X_valid = np.append(X_valid, patient_valid_set[0], axis=0)
        y_valid = np.append(y_valid, patient_valid_set[1], axis=0)

        X_test = np.append(X_test, patient_test_set[0], axis=0)
        y_test = np.append(y_test, patient_test_set[1], axis=0)

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]

def train():
    g = DataGenerator()
    X, y = g.data()
    train_set, valid_set, test_set = g.split(X, y)

    model_checkpoints_dirname = os.path.join(MODEL_DIR, 'ad_checkpoints', datetime.now().strftime('%Y_%m%d_%H%M_%S'))
    tensorboard_log_dirname = os.path.join(model_checkpoints_dirname, 'logs')
    os.makedirs(model_checkpoints_dirname)
    os.makedirs(tensorboard_log_dirname)

    # do normalize using means and stds from training data
    train_set[0], means_and_stds = DataGenerator.normalize(train_set[0])
    valid_set[0], _ = DataGenerator.normalize(valid_set[0], means_and_stds)
    test_set[0], _ = DataGenerator.normalize(test_set[0], means_and_stds)

    # save means and stds
    with open(model_checkpoints_dirname + '/means_and_stds.pl', 'wb') as f:
        pickle.dump(means_and_stds, f)

    model = get_model()
    model.summary()

    callbacks = [
        # EarlyStopping(patience=62),
        # ReduceLROnPlateau(patience=25, cooldown=5, verbose=1),
        ModelCheckpoint(model_checkpoints_dirname + '/{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True),
        # TensorBoard(log_dir=tensorboard_log_dirname),
        WandbCallback(log_gradients=True, training_data=train_set)
    ]

    print(1. - valid_set[1][:, 0].sum() / valid_set[1][:, 0].shape[0])
    print(1. - train_set[1][:, 0].sum() / train_set[1][:, 0].shape[0])

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=40, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)

    y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
    y_true = test_set[1][:, 1]

    print('Total accuracy:', sklearn.metrics.accuracy_score(y_true, y_pred))
    print(sklearn.metrics.classification_report(y_true, y_pred))

    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    train()
