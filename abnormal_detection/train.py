#!/usr/bin/env python3
import pickle
import numpy as np
import configparser
import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback
wandb.init(project='ekg-abnormal_detection', entity='toosyou')

import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ekg.callbacks import LogBest

from ekg.utils import data_utils
from ekg.utils.data_utils import BaseDataGenerator
import evaluation

from ekg.models.backbone import backbone

# read config
config = configparser.ConfigParser()
config.read('./config.cfg')

# search result
set_wandb_config({
    # model
    'sincconv_filter_length': 31,
    'sincconv_nfilters': 8,

    'branch_nlayers': 4,

    'ekg_kernel_length': 35,
    'hs_kernel_length': 13,

    'final_nlayers': 3,
    'final_kernel_length': 7,
    'final_nonlocal_nlayers': 0,

    'kernel_initializer': 'glorot_uniform',
    'skip_connection': False,
    'crop_center': True,

    # data
    'remove_dirty': 2, # deprecated, always remove dirty data
    'datasets': ['big_exam', 'audicor_10s'], # 'big_exam', 'audicor_10s'

    'big_exam_ekg_channels': [1], # [0, 1, 2, 3, 4, 5, 6, 7],
    'big_exam_hs_channels': [8, 9],

    'audicor_10s_ekg_channels': [0],
    'audicor_10s_hs_channels': [1],

    'downsample': 'direct', # average

}, include_preprocessing_setting=True)

set_wandb_config({
    'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
    'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
    'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
}, include_preprocessing_setting=False)

class DataGenerator(BaseDataGenerator):
    def __init__(self, config=config, wandb_config=wandb.config):
        super().__init__(big_exam_dir = config['Big_Exam']['output_dir'],
                            audicor_10s_dir = config['Audicor_10s']['output_dir'],
                            wandb_config = wandb_config)

    def get_abnormal_y(self):
        return np.ones((self.abnormal_X.shape[0], ))

    def get_normal_y(self):
        return np.zeros((self.normal_X.shape[0], ))

    def preprocessing(self):
        # make ys one-hot
        self.normal_y = keras.utils.to_categorical(self.normal_y, num_classes=2, dtype=np.int)
        self.abnormal_y = keras.utils.to_categorical(self.abnormal_y, num_classes=2, dtype=np.int)

def train():
    g = DataGenerator()
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2)
    model.compile(Adam(amsgrad=True), 'binary_crossentropy', metrics=['acc'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=40, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    from keras.models import load_model
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
    train()
