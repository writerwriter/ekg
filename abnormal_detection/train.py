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
from ekg.callbacks import GarbageCollector
from ekg.utils.data_utils import BaseDataGenerator
from ekg.utils.datasets import BigExamLoader, Audicor10sLoader
import evaluation
from evaluation import print_statistics

from ekg.models.backbone import backbone

def preprocessing(dataloader):
    # make ys one-hot
    dataloader.normal_y = keras.utils.to_categorical(dataloader.normal_y, num_classes=2, dtype=np.int)
    dataloader.abnormal_y = keras.utils.to_categorical(dataloader.abnormal_y, num_classes=2, dtype=np.int)

class AbnormalBigExamLoader(BigExamLoader):
    def load_abnormal_y(self):
        return np.ones((self.abnormal_X.shape[0], ))
    def load_normal_y(self):
        return np.zeros((self.normal_X.shape[0], ))

class AbnormalAudicor10sLoader(Audicor10sLoader):
    def load_abnormal_y(self):
        return np.ones((self.abnormal_X.shape[0], ))
    def load_normal_y(self):
        return np.zeros((self.normal_X.shape[0], ))

class AbnormalDataGenerator(BaseDataGenerator):
    @staticmethod
    def has_empty(datasets, threshold=2):
        return False

def train():
    dataloaders = list()
    if 'big_exam' in wandb.config.datasets:
        dataloaders.append(AbnormalBigExamLoader(wandb_config=wandb.config))
    if 'audicor_10s' in wandb.config.datasets:
        dataloaders.append(AbnormalAudicor10sLoader(wandb_config=wandb.config))

    g = AbnormalDataGenerator(  dataloaders=dataloaders,
                                wandb_config=wandb.config,
                                preprocessing_fn=preprocessing)
    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set)

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2)
    model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), 
                    'binary_crossentropy', metrics=['acc'])
    model.summary()
    wandb.log({'model_params': model.count_params()}, commit=False)

    callbacks = [
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        WandbCallback(),
        GarbageCollector(),
        EarlyStopping(monitor='val_loss', patience=50),
    ]

    model.fit(train_set[0], train_set[1], 
                batch_size=wandb.config.batch_size, epochs=200, 
                validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)
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

    evaluation.evaluation([model], test_set)

if __name__ == '__main__':
    wandb.init(project='ekg-abnormal_detection', entity='toosyou')

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

        'batch_size': 64,
        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': True,
        'se_block': True,

        'prediction_head': False,
        'include_info': False,

        'radam': True,

        'wavelet': False,
        'wavelet_scale_length': 25,

        # data
        'remove_dirty': 2, # deprecated, always remove dirty data
        'datasets': ['big_exam', 'audicor_10s'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [1], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8, 9],
        'big_exam_only_train': False,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,

        'downsample': 'direct', # average
        'with_normal_subjects': True,        # of course
        'normal_subjects_only_train': False, # of course

        'tf': 2.2

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()
