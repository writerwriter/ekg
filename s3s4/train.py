#!/usr/bin/env python3
import pickle
import numpy as np
import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_radam import RAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ekg.callbacks import LogBest

from ekg.utils import data_utils
from ekg.utils.data_utils import BaseDataGenerator
from ekg.utils.datasets import BigExamLoader, Audicor10sLoader
import evaluation
from evaluation import print_statistics

from ekg.models.backbone import backbone

def get_data():
    # load data
    big_signal_s3s4label_hs1 = np.load('./s3s4/big_signal_s3s4label_hs1.npy')
    big_signal_s3s4label_hs2 = np.load('./s3s4/big_signal_s3s4label_hs2.npy')

    filenames = big_signal_s3s4label_hs1[:, 0]
    s3_labels = big_signal_s3s4label_hs1[:, 1].astype(float).astype(int)
    s4_labels = big_signal_s3s4label_hs1[:, 2].astype(float).astype(int)

    hs1 = big_signal_s3s4label_hs1[:, 3:][:, :, np.newaxis].astype(int) # (?, signal_length, 1)
    hs2 = big_signal_s3s4label_hs2[:, 3:][:, :, np.newaxis].astype(int) # (?, signal_length, 1)

    signals = np.append(hs1, hs2, axis=-1) # (?, signal_length, 2)

    # preprocessing
    s3s4_labels = np.logical_or(s3_labels.astype(bool), s4_labels.astype(bool))
    s3s4_labels = keras.utils.to_categorical(s3s4_labels, num_classes=2, dtype=np.int) # to one hot
    subject_ids = np.vectorize(lambda s: s.split('/')[0])(filenames)

    # split data by subject ids
    train_set, valid_set, test_set = data_utils.subject_split(signals, s3s4_labels, subject_ids)

    # normalize
    train_set[0], means_and_stds = BaseDataGenerator.normalize(train_set[0])
    valid_set[0], _ = BaseDataGenerator.normalize(valid_set[0], means_and_stds)
    test_set[0], _ = BaseDataGenerator.normalize(test_set[0], means_and_stds)

    return train_set, valid_set, test_set, means_and_stds

def train():
    train_set, valid_set, test_set, means_and_stds = get_data()
    print_statistics(train_set, valid_set, test_set)

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2)
    model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), 
                    'binary_crossentropy', metrics=['acc'])
    model.summary()
    wandb.log({'model_params': model.count_params()}, commit=False)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=wandb.config.batch_size, 
                    epochs=200, validation_data=(valid_set[0], valid_set[1]), 
                    callbacks=callbacks, shuffle=True, class_weight={   0: train_set[1][:, 1].sum(), 
                                                                        1: train_set[1][:, 0].sum()})
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    from tensorflow.keras.models import load_model
    from ekg.layers import LeftCropLike, CenterCropLike
    from ekg.layers.sincnet import SincConv1D

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)

    evaluation.evaluation(model, test_set)

if __name__ == '__main__':
    wandb.init(project='ekg-s3s4', entity='toosyou')

    # search result
    set_wandb_config({
        # model
        'sincconv_filter_length': 31,
        'sincconv_nfilters': 8,

        'branch_nlayers': 1,

        'ekg_kernel_length': 35,
        'hs_kernel_length': 35,

        'ekg_nfilters': 1,
        'hs_nfilters': 1,

        'final_nlayers': 5,
        'final_kernel_length': 21,
        'final_nonlocal_nlayers': 0,
        'final_nfilters': 8,

        'prediction_head': False,
        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,
        'prediction_nfilters': 8,

        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': False,

        'batch_size': 64,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': True,
        'se_block': True,

        'include_info': False, # only works with audicor_10s
        'infos': ['sex', 'age', 'height', 'weight', 'BMI'],
        'info_apply_noise': True,
        'info_noise_stds': [0, 1, 1, 1, 0.25], # stds of gaussian noise
        'info_nlayers': 2,
        'info_units': 8,

        'radam': True,

        # data
        'remove_dirty': 2, # deprecated, always remove dirty data
        'datasets': ['big_exam'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8, 9],
        'big_exam_only_train': False,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,

        'downsample': 'direct', # average

        'tf': 2.2

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
