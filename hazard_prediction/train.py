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
from ekg.losses import AFTLoss

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
from ekg.utils.data_utils import BaseDataGenerator
from ekg.callbacks import LogBest, ConcordanceIndex

from ekg.models.backbone import backbone
from ekg.losses import negative_hazard_log_likelihood

from evaluation import evaluation_plot, print_statistics
from evaluation import evaluation, to_prediction_model

from sklearn.utils import shuffle
from tensorflow import keras

from tensorflow.keras.models import load_model
from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D

def to_cs_st(y):
    cs = y[:, :, 0]
    st = y[:, :, 1]
    return st * (-1 * (cs == 0) + 1 * (cs == 1))

class LossChecker(keras.callbacks.Callback):
    def __init__(self, train_set, valid_set):
        self.train_set = train_set
        self.valid_set = valid_set

    def loss(self, cs_st, pred_risk):

        cs = (cs_st > 0).astype(float) # (?, n_events)
        st = np.abs(cs_st) # (?, n_events)

        total_loss = 0
        for i in range(cs.shape[-1]): # for every event
            this_cs = cs[:, i]
            this_st = st[:, i]
            this_risk = pred_risk[:, i]

            sorting_indices = np.argsort(this_st)[::-1]
            sorted_cs = this_cs[sorting_indices]
            sorted_risk = this_risk[sorting_indices]

            hazard_ratio = np.exp(sorted_risk)
            log_risk = np.log(np.cumsum(hazard_ratio))
            uncensored_likelihood = sorted_risk - log_risk
            censored_likelihood = sorted_cs * uncensored_likelihood
            total_loss += -np.sum(censored_likelihood)
            if total_loss != total_loss:
                import sys; sys.exit(-1)
                import pdb; pdb.set_trace()
        return total_loss / cs.shape[-1]

    def on_epoch_end(self, epoch, logs={}):
        train_pred = self.model.predict(self.train_set[0])
        valid_pred = self.model.predict(self.valid_set[0])

        print()
        print('training loss:', self.loss(to_cs_st(self.train_set[1]), train_pred))
        print('validation loss:', self.loss(to_cs_st(self.valid_set[1]), valid_pred))

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
    if wandb.config.include_info:
        return Model([prediction_model.input[0], prediction_model.input[1], cs_input, st_input], output)
    else:
        return Model([prediction_model.input, cs_input, st_input], output)

def get_train_valid(train_set, valid_set):
    if wandb.config.loss == 'AFT': # Note: doesn't work with info
        if wandb.config.include_info:
            X_train = train_set[0].copy()
            X_train['cs_input'] = train_set[1][:, :, 0]
            X_train['st_input'] = train_set[1][:, :, 1]

            X_valid = valid_set[0].copy()
            X_valid['cs_input'] = valid_set[1][:, :, 0]
            X_valid['st_input'] = valid_set[1][:, :, 1]
        else:
            X_train = [train_set[0], train_set[1][:, :, 0], train_set[1][:, :, 1]]
            X_valid = [valid_set[0], valid_set[1][:, :, 0], valid_set[1][:, :, 1]]

        y_train, y_valid = None, None
    else:
        X_train, X_valid = train_set[0], valid_set[0]
        y_train, y_valid = to_cs_st(train_set[1]), to_cs_st(valid_set[1])

    return X_train, y_train, X_valid, y_valid

def train():
    dataloaders = list()
    if 'big_exam' in wandb.config.datasets:
        dataloaders.append(HazardBigExamLoader(wandb_config=wandb.config))
    if 'audicor_10s' in wandb.config.datasets:
        dataloaders.append(HazardAudicor10sLoader(wandb_config=wandb.config))

    g = BaseDataGenerator(dataloaders=dataloaders,
                            wandb_config=wandb.config,
                            preprocessing_fn=preprocessing)

    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set, wandb.config.events)

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    prediction_model = backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
    trainable_model = get_trainable_model(prediction_model, AFTLoss(len(wandb.config.events), name='AFT_loss')) if wandb.config.loss == 'AFT' else prediction_model

    loss = None if wandb.config.loss == 'AFT' else negative_hazard_log_likelihood(wandb.config.event_weights,
                                                    wandb.config.output_l1_regularizer,
                                                    wandb.config.output_l2_regularizer)

    trainable_model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), loss=loss)
    trainable_model.summary()
    wandb.log({'model_params': trainable_model.count_params()}, commit=False)

    c_index_reverse = (wandb.config.loss != 'AFT')

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        # LossChecker(train_set, valid_set),
        ConcordanceIndex(train_set, valid_set, wandb.config.events, prediction_model, reverse=c_index_reverse),
        LogBest(records=['val_loss', 'loss'] + 
                    ['{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['val_{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['{}_std'.format(event_name) for event_name in wandb.config.events]),
        WandbCallback(),
        EarlyStopping(monitor='val_loss', patience=50), # must be placed last otherwise it won't work
    ]

    if wandb.config.loss == 'AFT': callbacks = [LossVariableChecker(wandb.config.events)] + callbacks

    X_train, y_train, X_valid, y_valid = get_train_valid(train_set, valid_set)
    trainable_model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=500,
                validation_data=(X_valid, y_valid), callbacks=callbacks, shuffle=True)
    trainable_model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike,
        'AFTLoss': AFTLoss,
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)
    prediction_model = to_prediction_model(model, wandb.config.include_info) if wandb.config.loss == 'AFT' else model

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
        'sincconv_filter_length': 121,
        'sincconv_nfilters': 32,

        'branch_nlayers': 1,

        'ekg_kernel_length': 13,
        'hs_kernel_length': 35,

        'ekg_nfilters': 1,
        'hs_nfilters': 1,

        'final_nlayers': 2, # 4
        'final_kernel_length': 5,
        'final_nonlocal_nlayers': 0,
        'final_nfilters': 8,

        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,
        'prediction_nfilters': 8,

        'batch_size': 64,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': True,
        'crop_center': True,
        'se_block': True,

        'prediction_head': True,
        
        'include_info': False, # only works with audicor_10s
        'infos': ['sex', 'age', 'height', 'weight', 'BMI'],
        'info_apply_noise': True,
        'info_noise_stds': [0, 1, 1, 1, 0.25], # stds of gaussian noise
        'info_nlayers': 2,
        'info_units': 8,

        'radam': True,

        'loss': 'AFT',

        # data
        'events': ['ADHF'], # 'MI', 'Stroke', 'CVD', 'Mortality'
        'event_weights': [1],
        'censoring_limit': 400, # 99999 if no limit specified

        'output_l1_regularizer': 0, # 0 if disable
        'output_l2_regularizer': 0, # 0 if disable # 0.01 - 0.1

        'datasets': ['audicor_10s'], # 'big_exam', 'audicor_10s'

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
