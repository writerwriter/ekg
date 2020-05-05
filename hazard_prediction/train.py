#!/usr/bin/env python3
import pickle
import better_exceptions; better_exceptions.hook()

import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.optimizers import Adam
from keras_radam import RAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataloader import HazardBigExamLoader, HazardAudicor10sLoader
from dataloader import preprocessing
from ekg.utils import data_utils
from ekg.utils.data_utils import BaseDataGenerator
from ekg.callbacks import LogBest, ConcordanceIndex, VarianceChecker

from ekg.models.backbone import backbone
from ekg.losses import negative_hazard_log_likelihood

from evaluation import evaluation_plot, print_statistics
from evaluation import evaluation, to_prediction_model

from tensorflow.keras.models import load_model
from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D

from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from ekg.losses.multi_task import MultiHazardLoss

def to_cs_st(y):
    cs = y[:, :, 0]
    st = y[:, :, 1]
    return st * (-1 * (cs == 0) + 1 * (cs == 1))

def get_multi_task_model(prediction_model):
    cs_input = Input(shape=(len(wandb.config.events, )), name='cs_input') # (?, n_events)
    st_input = Input(shape=(len(wandb.config.events, )), name='st_input') # (?, n_events)
    risks = prediction_model.layers[-1].output # (?, n_events)

    loss_inputs = list()
    for i, event in enumerate(wandb.config.events):
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='cs_{}'.format(event))(cs_input)) # cs
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='st_{}'.format(event))(st_input)) # st
        
    for i, event in enumerate(wandb.config.events):
        loss_inputs.append(Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='risk_{}'.format(event))(risks)) # risk

    output = MultiHazardLoss(n_outputs = len(wandb.config.events))(loss_inputs)
    return Model([prediction_model.input, cs_input, st_input], output)

def get_train_valid(train_set, valid_set):
    if wandb.config.multi_task_loss:
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

    # get model
    prediction_model = backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
    trainable_model = get_multi_task_model(prediction_model) if wandb.config.multi_task_loss else prediction_model

    loss = None if wandb.config.multi_task_loss else negative_hazard_log_likelihood(wandb.config.event_weights,
                                                    wandb.config.output_l1_regularizer,
                                                    wandb.config.output_l2_regularizer)
                                            
    trainable_model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), loss=loss)
    trainable_model.summary()
    wandb.log({'model_params': trainable_model.count_params()}, commit=False)

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        # LossChecker(train_set, valid_set),
        ConcordanceIndex(train_set, valid_set, wandb.config.events, prediction_model),
        LogBest(records=['val_loss', 'loss'] + 
                    ['{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['val_{}_cindex'.format(event_name) for event_name in wandb.config.events]),
        WandbCallback(),
        EarlyStopping(monitor='val_loss', patience=50), # must be placed last otherwise it won't work
    ]

    if wandb.config.multi_task_loss:
        callbacks = [VarianceChecker(wandb.config.events)] + callbacks

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
        'MultiHazardLoss': MultiHazardLoss,
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)
    prediction_model = to_prediction_model(model)

    print('Training set:')
    evaluation(prediction_model, train_set, wandb.config.events)

    print('Testing set:')
    evaluation(prediction_model, test_set, wandb.config.events)

    evaluation_plot(prediction_model, train_set, train_set, 'training - ')
    evaluation_plot(prediction_model, train_set, valid_set, 'validation - ')
    evaluation_plot(prediction_model, train_set, test_set, 'testing - ')

if __name__ == '__main__':
    wandb.init(project='ekg-hazard_prediction', entity='toosyou')

    # search result
    set_wandb_config({
        'sincconv_filter_length': 121,
        'sincconv_nfilters': 32,

        'branch_nlayers': 1,

        'ekg_kernel_length': 35,
        'hs_kernel_length': 35,

        'ekg_nfilters': 8,
        'hs_nfilters': 8,

        'final_nlayers': 7,
        'final_kernel_length': 13,
        'final_nonlocal_nlayers': 0,

        'batch_size': 128,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': True,

        'radam': True,

        'multi_task_loss': True,

        # data
        'events': ['ADHF', 'Mortality', 'MI', 'Stroke', 'CVD'], # 'MI', 'Stroke', 'CVD'
        'event_weights': [1, 1, 1, 1, 1],
        'censoring_limit': 99999, # 99999 if no limit specified

        'output_l1_regularizer': 0, # 0 if disable
        'output_l2_regularizer': 0, # 0 if disable # 0.01 - 0.1

        'remove_dirty': 2, # deprecated, always remove dirty data
        'datasets': ['big_exam'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [0, 1, 2, 3, 4, 5, 6, 7], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8, 9],
        'big_exam_only_train': False,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,

        'downsample': 'direct', # average

        'tf': '2.2',

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
