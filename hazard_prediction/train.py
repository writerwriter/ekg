#!/usr/bin/env python3
import pickle
import better_exceptions; better_exceptions.hook()

import matplotlib.pyplot as plt
import numpy as np
import configparser
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

def to_trainable_X(dataset):
    X, y = dataset[0].copy(), dataset[1]
    X['cs_input'], X['st_input'] = y[..., 0], y[..., 1]
    return X

def get_loss_layer(loss):
    return {
        'aft': AFTLoss(len(wandb.config.events), 
                        wandb.config.AFT_distribution, 
                        wandb.config.AFT_initial_sigma,
                        name='AFT_loss'),
        'cox': CoxLoss(len(wandb.config.events), wandb.config.event_weights, name='Cox_loss')
    }[loss.lower()]

def dataset_shuffle(dataset):
    _, value = list(dataset.items())[0]
    shuffle_indices = np.arange(value.shape[0])
    np.random.shuffle(shuffle_indices)
    
    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

class HazardDataGenerator(BaseDataGenerator):
    @staticmethod
    def has_empty(datasets, threshold=2):
        for dataset in datasets:
            y = dataset[1]
            for i in range(y.shape[1]):
                if (y[:, i, 0] == 1).sum() <= threshold: # number of signals with events
                    return True
        return False

def train():
    dataloaders = list()
    if 'big_exam' in wandb.config.datasets:
        dataloaders.append(HazardBigExamLoader(wandb_config=wandb.config))
    if 'audicor_10s' in wandb.config.datasets:
        dataloaders.append(HazardAudicor10sLoader(wandb_config=wandb.config))

    g = HazardDataGenerator(  dataloaders=dataloaders,
                            wandb_config=wandb.config,
                            preprocessing_fn=preprocessing)

    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set, wandb.config.events)

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    if wandb.config.include_info and wandb.config.info_apply_noise:
        wandb.config.info_norm_noise_std = np.array(wandb.config.info_noise_stds) / np.array(g.means_and_stds[1][1])

    prediction_model = backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
    trainable_model = get_trainable_model(prediction_model, get_loss_layer(wandb.config.loss))

    trainable_model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), loss=None)
    trainable_model.summary()
    wandb.log({'model_params': trainable_model.count_params()}, commit=False)

    c_index_reverse, scatter_exp = (wandb.config.loss != 'AFT'), (wandb.config.loss == 'AFT')
    scatter_xlabel = 'predicted survival time (days)'if wandb.config.loss == 'AFT' else 'predicted risk'

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LossVariableChecker(wandb.config.events),
        ConcordanceIndex(train_set, valid_set, wandb.config.events, prediction_model, reverse=c_index_reverse),
        LogBest(records=['val_loss', 'loss'] + 
                    ['{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['val_{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['{}_sigma'.format(event_name) for event_name in wandb.config.events]),
        WandbCallback(),
        EarlyStopping(monitor='val_loss', patience=50), # must be placed last otherwise it won't work
    ]

    X_train, y_train, X_valid, y_valid = to_trainable_X(train_set), None, to_trainable_X(valid_set), None
    dataset_shuffle(X_train)
    dataset_shuffle(X_valid)
    
    trainable_model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=1000, 
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

    evaluation_plot(prediction_model, train_set, train_set, 'training - ', reverse=c_index_reverse, scatter_exp=scatter_exp, scatter_xlabel=scatter_xlabel)
    evaluation_plot(prediction_model, train_set, valid_set, 'validation - ', reverse=c_index_reverse, scatter_exp=scatter_exp, scatter_xlabel=scatter_xlabel)
    evaluation_plot(prediction_model, train_set, test_set, 'testing - ', reverse=c_index_reverse, scatter_exp=scatter_exp, scatter_xlabel=scatter_xlabel)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '../config.cfg'))

    wandb.init(project='ekg-hazard_prediction', entity=config['General']['wandb_entity'])

    # search result
    set_wandb_config({
        'sincconv_filter_length': 26,
        'sincconv_nfilters': 8,

        'branch_nlayers': 1,

        'ekg_kernel_length': 35,
        'hs_kernel_length': 21,

        'ekg_nfilters': 8,
        'hs_nfilters': 8,

        'final_nlayers': 3,
        'final_kernel_length': 5,
        'final_nonlocal_nlayers': 0,
        'final_nfilters': 8,

        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,
        'prediction_nfilters': 8,

        'batch_size': 64,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': True,
        'se_block': False,

        'prediction_head': False,
        
        'include_info': True,
        'infos': ['sex', 'age', 'height', 'weight', 'BMI'],
        'info_apply_noise': True,
        'info_noise_stds': [0, 1, 1, 1, 0.25], # , 1, 1, 0.25 # stds of gaussian noise
        'info_nlayers': 5,
        'info_units': 64,

        'radam': False,

        'loss': 'AFT', # Cox, AFT
        'AFT_distribution': 'log-logistic', # weibull, log-logistic
        'AFT_initial_sigma': 0.5,

        'wavelet': False,
        'wavelet_scale_length': 25,

        # data
        'events': ['ADHF', 'Mortality'], # 'MI', 'Stroke', 'CVD', 'Mortality'
        'event_weights': [1, 1, 1, 1],
        'censoring_limit': 99999, # 99999 if no limit specified

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
        'normal_subjects_only_train': False,

        'tf': '2.2',
        'remove_dirty': 2, # deprecated, always remove dirty data

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
