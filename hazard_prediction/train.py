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
from ekg.callbacks import LogBest, ConcordanceIndex

from ekg.models.backbone import backbone
from ekg.losses import negative_hazard_log_likelihood

from evaluation import evaluation_plot, print_statistics
from evaluation import evaluation

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

    model = backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
    model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), 
                    loss=negative_hazard_log_likelihood(wandb.config.event_weights,
                                                            wandb.config.output_l1_regularizer,
                                                            wandb.config.output_l2_regularizer))
    model.summary()
    wandb.log({'model_params': model.count_params()}, commit=False)

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LossChecker(train_set, valid_set),
        ConcordanceIndex(train_set, valid_set, wandb.config.events),
        LogBest(records=['val_loss', 'loss'] + 
                    ['{}_cindex'.format(event_name) for event_name in wandb.config.events] +
                    ['val_{}_cindex'.format(event_name) for event_name in wandb.config.events]),
        WandbCallback(log_gradients=False, training_data=train_set),
        EarlyStopping(monitor='val_loss', patience=20), # must be placed last otherwise it won't work
    ]

    train_set = shuffle(train_set[0], train_set[1])
    valid_set = shuffle(valid_set[0], valid_set[1])
    model.fit(train_set[0], to_cs_st(train_set[1]), batch_size=wandb.config.batch_size, epochs=500,
                validation_data=(valid_set[0], to_cs_st(valid_set[1])),
                callbacks=callbacks, shuffle=True)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)

    print('Training set:')
    evaluation(model, train_set, wandb.config.events)

    print('Testing set:')
    evaluation(model, test_set, wandb.config.events)

    evaluation_plot(model, train_set, train_set, 'training - ')
    evaluation_plot(model, train_set, valid_set, 'validation - ')
    evaluation_plot(model, train_set, test_set, 'testing - ')

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

        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,

        'batch_size': 128,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': True,
        'crop_center': True,

        'prediction_head': True,

        'radam': True,

        # data
        'events': ['ADHF', 'Mortality'], # 'MI', 'Stroke', 'CVD'
        'event_weights': [1, 1],
        'censoring_limit': 400, # 99999 if no limit specified

        'output_l1_regularizer': 0, # 0 if disable
        'output_l2_regularizer': 0, # 0 if disable # 0.01 - 0.1

        'remove_dirty': 2, # deprecated, always remove dirty data
        'datasets': ['big_exam', 'audicor_10s'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [1], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8, 9],
        'big_exam_only_train': True,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,

        'downsample': 'direct', # average
        'with_normal_subjects': False,

        'tf': '2.2',

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
