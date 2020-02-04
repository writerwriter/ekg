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
wandb.init(project='ekg-abnormal_detection', entity='toosyou')

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

    'branch_nlayers': 4,

    'ekg_kernel_length': 35,
    'hs_kernel_length': 13,

    'final_nlayers': 3,
    'final_kernel_length': 7,
    'final_nonlocal_nlayers': 0,

    'kernel_initializer': 'glorot_uniform',
    'skip_connection': False,
    'crop_center': True,

    'remove_dirty': 2,

    'n_ekg_channels': 0,
    'n_hs_channels': 2,
})

def train():
    g = DataGenerator(remove_dirty=wandb.config.remove_dirty, 
                        n_ekg_channels=wandb.config.n_ekg_channels, 
                        n_hs_channels=wandb.config.n_hs_channels)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2, 
                        n_ekg_channels=wandb.config.n_ekg_channels,
                        n_hs_channels=wandb.config.n_hs_channels)
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
    evaluation(model, test_set)

if __name__ == '__main__':
    train()
