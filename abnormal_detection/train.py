#!/usr/bin/env python3
import numpy as np
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
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D, ReLU, Bidirectional, Maximum
from keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate, Flatten, Dropout, LSTM, Reshape, SeparableConv1D
from keras.utils import conv_utils

from data_generator import DataGenerator
from ekg.layers import LeftCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.layers.non_local import non_local_block
from ekg.utils.eval_utils import print_cm
from eval import evaluation

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import wandb
from wandb.keras import WandbCallback
wandb.init(name='remove_dirty2_best_param', project='ekg-abnormal_detection', entity='toosyou')

def set_wandb_config(params, overwrite=False):
    for key, value in params.items():
        if key not in wandb.config._items or overwrite:
            wandb.config.update({key: value})

# search result
set_wandb_config({
    'sincconv_filter_length': 31,
    'sincconv_nfilters': 8,

    'branch_nlayers': 1,

    'ekg_kernel_length': 21,
    'hs_kernel_length': 5,

    'final_nlayers': 5,
    'final_kernel_length': 13,
    'final_nonlocal_nlayers': 0,

    'remove_dirty': 2,
})

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_model():
    def heart_sound_branch(hs):
        sincconv_filter_length = wandb.config.sincconv_filter_length - (wandb.config.sincconv_filter_length+1) % 2
        hs = SincConv1D(wandb.config.sincconv_nfilters, sincconv_filter_length, 1000)(hs)
        hs = BatchNormalization()(hs)

        for _ in range(wandb.config.branch_nlayers):
            hs = Conv1D(8, wandb.config.hs_kernel_length, activation='relu', padding='same')(hs)
            hs = BatchNormalization()(hs)
            hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)
        return hs

    # total_input = Input((10000, 10))
    total_input = Input((None, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    # ekg branch
    ekg = ekg_input
    for _ in range(wandb.config.branch_nlayers):
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
            output = MaxPooling1D(2, padding='same')(output)

    output = GlobalAveragePooling1D()(output)
    output = Dense(2, activation='softmax')(output)

    model = Model(total_input, output)
    model.compile(Adam(amsgrad=True), 'binary_crossentropy', metrics=['acc'])
    return model

class LogBest(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = -1
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if self.best_val_loss > logs.get('val_loss'): # update
            self.best_val_loss = logs.get('val_loss')
            self.best_val_acc = logs.get('val_acc')
            self.best_loss = logs.get('loss')
            self.best_acc = logs.get('acc')
            self.best_epoch = epoch

            wandb.log({
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'best_loss': self.best_loss,
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch
            }, commit=False)

def train():
    g = DataGenerator(remove_dirty=wandb.config.remove_dirty,
                        ekg_scaling_prob=wandb.config.ekg_scaling_prob,
                        hs_scaling_prob=wandb.config.hs_scaling_prob,
                        time_stretch_prob=wandb.config.time_stretch_prob)
    train_set, valid_set, test_set = g.get()

    model_checkpoints_dirname = os.path.join(MODEL_DIR, 'ad_checkpoints', datetime.now().strftime('%Y_%m%d_%H%M_%S'))
    tensorboard_log_dirname = os.path.join(model_checkpoints_dirname, 'logs')
    os.makedirs(model_checkpoints_dirname)
    os.makedirs(tensorboard_log_dirname)

    # save means and stds
    with open(model_checkpoints_dirname + '/means_and_stds.pl', 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = get_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        # ModelCheckpoint(model_checkpoints_dirname + '/{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True),
        # TensorBoard(log_dir=tensorboard_log_dirname),
        LogBest(),
        WandbCallback(log_gradients=True, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=40, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)
    evaluation(model, test_set)

if __name__ == '__main__':
    train()
