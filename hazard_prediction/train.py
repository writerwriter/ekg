#!/usr/bin/env python3
import pickle
import better_exceptions; better_exceptions.hook()

import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config
from ekg.callbacks import LossVariableChecker
from ekg.losses import AFTLoss, CoxLoss

# for loging result
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from keras_radam import RAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataloader import HazardBigExamLoader, HazardAudicor10sLoader
from dataloader import preprocessing
from ekg.utils import data_utils
from ekg.utils.data_utils import BaseDataGenerator, generate_wavelet
from ekg.callbacks import LogBest, ConcordanceIndex

from ekg.models.backbone import backbone

from evaluation import evaluation_plot, print_statistics
from evaluation import evaluation, to_prediction_model

from sklearn.utils import shuffle
from tensorflow import keras

from tensorflow.keras.models import load_model
from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D

def get_trainable_model(prediction_model, loss_layer):
    cs_input = Input(shape=(len(wandb.config.events, )), name='cs_input') # (?, n_events)
    st_input = Input(shape=(len(wandb.config.events, )), name='st_input') # (?, n_events)
    risks = prediction_model.layers[-1].output # (?, n_events)

    loss_inputs = list()
    for i, event in enumerate(wandb.config.events):
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='cs_{}'.format(event))(cs_input)) # cs
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='st_{}'.format(event))(st_input)) # st
        
    for i, event in enumerate(wandb.config.events):
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='risk_{}'.format(event))(risks)) # risk

    output = loss_layer(loss_inputs)
    if isinstance(prediction_model.input, list):
        return Model(prediction_model.input + [cs_input, st_input], output)
    else:
        return Model([prediction_model.input, cs_input, st_input], output)

class HazardDataGenerator(BaseDataGenerator):
    def __init__(self, do_wavelet, **kwargs):
        self.do_wavelet = do_wavelet
        super().__init__(**kwargs)

    def get(self):
        train_set, valid_set, test_set = super().get()

        if self.do_wavelet:
            for X in [train_set[0], valid_set[0], test_set[0]]:
                if self.config.n_ekg_channels != 0:
                    X['ekg_input'] = X['ekg_hs_input'][..., :self.config.n_ekg_channels]
                if self.config.n_hs_channels != 0:
                    hs = X['ekg_hs_input'][..., -self.config.n_hs_channels:] # (?, n_samples, n_channels)
                    X['hs_input'] = data_utils.mp_generate_wavelet(hs, 
                                                                    wandb.config.sampling_rate, 
                                                                    self.config.wavelet_scale_length,
                                                                    'Generate Wavelets')
                X.pop('ekg_hs_input')
        return train_set, valid_set, test_set

def to_trainable_X(dataset):
    X, y = dataset[0].copy(), dataset[1]
    X['cs_input'], X['st_input'] = y[..., 0], y[..., 1]
    return X

class HazardSequence(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, do_wavelet):
        self.X, self.y, self.batch_size, self.do_wavelet = X, y, batch_size, do_wavelet

    def n_instances(self):
        _, value = list(self.X.items())[0] # get the first input
        return value.shape[0]

    def __len__(self):
        return np.ceil(self.n_instances() / self.batch_size).astype(int)

    @staticmethod
    def __move_label_to_X(batch_X, batch_y):
        batch_X['cs_input'] = batch_y[..., 0]
        batch_X['st_input'] = batch_y[..., 1]
        return batch_X

    def __getitem__(self, index):
        batch_X = dict()

        batch_slice = np.s_[index * self.batch_size: min((index + 1) * self.batch_size, self.n_instances())]
        for key, value in self.X.items():
            batch_X[key] = value[batch_slice]

        if self.do_wavelet:
            if wandb.config.n_ekg_channels != 0:
                batch_X['ekg_input'] = batch_X['ekg_hs_input'][..., :wandb.config.n_ekg_channels]
            if wandb.config.n_hs_channels != 0:
                hs = batch_X['ekg_hs_input'][..., -wandb.config.n_hs_channels:] # (?, n_samples, n_channels)
                wavelet = np.zeros((hs.shape[0], 
                                    wandb.config.wavelet_scale_length, 
                                    hs.shape[1], wandb.config.n_hs_channels)) # (?, scale_length, n_samples, n_channels)
                for i in range(hs.shape[0]):
                    for j in range(hs.shape[-1]):
                        wavelet[i, :, :, j] = generate_wavelet(hs[i, :, j], 
                                                                wandb.config.sampling_rate, 
                                                                wandb.config.wavelet_scale_length)
                batch_X['hs_input'] = wavelet
            batch_X.pop('ekg_hs_input')

        if self.y is not None:
            batch_y = self.y[batch_slice]
            batch_X = self.__move_label_to_X(batch_X, batch_y)

        return batch_X, None

    def on_epoch_end(self):
        # shuffle
        shuffle_indices = shuffle(np.arange(self.n_instances()))
        for key, value in self.X.items():
            self.X[key] = value[shuffle_indices]

        if self.y is not None:
            self.y = self.y[shuffle_indices]

def get_loss_layer(loss):
    return {
        'aft': AFTLoss(len(wandb.config.events), name='AFT_loss'),
        'cox': CoxLoss(len(wandb.config.events), wandb.config.event_weights, name='Cox_loss')
    }[loss.lower()]

def train():
    dataloaders = list()
    if 'big_exam' in wandb.config.datasets:
        dataloaders.append(HazardBigExamLoader(wandb_config=wandb.config))
    if 'audicor_10s' in wandb.config.datasets:
        dataloaders.append(HazardAudicor10sLoader(wandb_config=wandb.config))

    g = HazardDataGenerator(do_wavelet=wandb.config.wavelet,
                            dataloaders=dataloaders,
                            wandb_config=wandb.config,
                            preprocessing_fn=preprocessing)

    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set, wandb.config.events)

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    prediction_model = backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
    trainable_model = get_trainable_model(prediction_model, get_loss_layer(wandb.config.loss))

    trainable_model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), loss=None)
    trainable_model.summary()
    wandb.log({'model_params': trainable_model.count_params()}, commit=False)

    c_index_reverse = (wandb.config.loss != 'AFT')

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LossVariableChecker(wandb.config.events),
        ConcordanceIndex(train_set, valid_set, wandb.config.events, prediction_model, reverse=c_index_reverse),
        LogBest(records=['val_loss', 'loss'] + 
                    ['{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['val_{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['{}_std'.format(event_name) for event_name in wandb.config.events]),
        WandbCallback(),
        EarlyStopping(monitor='val_loss', patience=50), # must be placed last otherwise it won't work
    ]

    X_train, y_train, X_valid, y_valid = to_trainable_X(train_set), None, to_trainable_X(valid_set), None
    trainable_model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=500, 
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks, shuffle=True)
    trainable_model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike,
        'AFTLoss': AFTLoss,
        'CoxLoss': CoxLoss,
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)
    prediction_model = to_prediction_model(model, wandb.config.include_info)

    print('Training set:')
    evaluation(prediction_model, train_set, wandb.config.events, reverse=c_index_reverse)

    print('Testing set:')
    evaluation(prediction_model, test_set, wandb.config.events, reverse=c_index_reverse)

    evaluation_plot(prediction_model, train_set, train_set, 'training - ', reverse=c_index_reverse)
    evaluation_plot(prediction_model, train_set, valid_set, 'validation - ', reverse=c_index_reverse)
    evaluation_plot(prediction_model, train_set, test_set, 'testing - ', reverse=c_index_reverse)

if __name__ == '__main__':
    wandb.init(project='ekg-hazard_prediction', entity='toosyou')

    # search result
    set_wandb_config({
        'sincconv_filter_length': 63,
        'sincconv_nfilters': 8,

        'branch_nlayers': 2,

        'ekg_kernel_length': 13,
        'hs_kernel_length': 35,

        'ekg_nfilters': 2,
        'hs_nfilters': 2,

        'final_nlayers': 5,
        'final_kernel_length': 5,
        'final_nonlocal_nlayers': 0,
        'final_nfilters': 8,

        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,
        'prediction_nfilters': 8,

        'batch_size': 32,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': True,
        'se_block': True,

        'prediction_head': False,
        
        'include_info': False, # only works with audicor_10s
        'infos': ['sex', 'age', 'height', 'weight', 'BMI'],
        'info_apply_noise': True,
        'info_noise_stds': [0, 1, 1, 1, 0.25], # stds of gaussian noise
        'info_nlayers': 2,
        'info_units': 8,

        'radam': True,

        'loss': 'AFT',

        'wavelet': True,
        'wavelet_scale_length': 25,
        'wavelet_nfilters': 16,

        # data
        'events': ['ADHF'], # 'MI', 'Stroke', 'CVD', 'Mortality'
        'event_weights': [1],
        'censoring_limit': 400, # 99999 if no limit specified

        'output_l1_regularizer': 0, # 0 if disable
        'output_l2_regularizer': 0, # 0 if disable # 0.01 - 0.1

        'datasets': ['big_exam', 'audicor_10s'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [1], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8, 9],
        'big_exam_only_train': False,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,
        'audicor_10s_ignore_888': True,

        'downsample': 'direct', # average
        'with_normal_subjects': False,
        'normal_subjects_only_train': True,

        'tf': '2.2',
        'remove_dirty': 2, # deprecated, always remove dirty data

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
