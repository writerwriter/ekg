#!/usr/bin/env python3
import pickle
import better_exceptions; better_exceptions.hook()

import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback
wandb.init(project='ekg-hazard_prediction', entity='toosyou')

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_generator import DataGenerator
# from eval import evaluation
from ekg.callbacks import LogBest, ConcordanceIndex

from ekg.models.backbone import backbone
from ekg.losses import negative_hazard_log_likelihood
from ekg.metrics import cindex

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
})

def to_cs_st(y):
    cs = y[:, :, 0]
    st = y[:, :, 1]
    return st * (-1 * (cs == 0) + 1 * (cs == 1))

def train():
    event_names = ['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality']
    g = DataGenerator(remove_dirty=wandb.config.remove_dirty, event_names=event_names)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=False, classes=len(event_names))
    model.compile(Adam(amsgrad=True), loss=negative_hazard_log_likelihood) # , metrics=[cindex])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        ConcordanceIndex(train_set, valid_set, event_names),
        LogBest(),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], to_cs_st(train_set[1]), batch_size=64, epochs=40, validation_data=(valid_set[0], to_cs_st(valid_set[1])), callbacks=callbacks, shuffle=True)
    '''
    batch_size = 64
    valid_set = g.shuffle(valid_set[0], valid_set[1], batch_size) # must be shuffled and sorted before training
    model.fit_generator(g.batch_generator(train_set[0], train_set[1], batch_size, shuffle=True, data_augmentation=False),
                        steps_per_epoch=int(np.ceil(train_set[0].shape[0] / batch_size)),
                        epochs=100, callbacks=callbacks,
                        validation_data=(valid_set[0], valid_set[1]),
                        shuffle=False)
    '''
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))
    # evaluation(model, test_set)

if __name__ == '__main__':
    train()
