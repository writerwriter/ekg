#!/usr/bin/env python3
import pickle
import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback
wandb.init(name='remove_dirty2_best_param_test', project='ekg-abnormal_detection', entity='toosyou')

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_generator import DataGenerator
from eval import evaluation
from ekg.callbacks import LogBest

from ekg.models.backbone import backbone

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

def train():
    g = DataGenerator(remove_dirty=wandb.config.remove_dirty)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2)
    model.compile(Adam(amsgrad=True), 'binary_crossentropy', metrics=['acc'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        WandbCallback(log_gradients=True, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=40, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)
    evaluation(model, test_set)

if __name__ == '__main__':
    train()
