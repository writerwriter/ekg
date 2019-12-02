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
from keras_radam import RAdam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_generator import DataGenerator
from eval import evaluation
from ekg.callbacks import LogBest, ConcordanceIndex

from ekg.models.backbone import backbone
from ekg.losses import negative_hazard_log_likelihood

from sklearn.utils import shuffle
import keras

# search result
set_wandb_config({
    'sincconv_filter_length': 31,
    'sincconv_nfilters': 8,

    'branch_nlayers': 3,

    'ekg_kernel_length': 21,
    'hs_kernel_length': 7,

    'final_nlayers': 3,
    'final_kernel_length': 13,
    'final_nonlocal_nlayers': 0,

    'kernel_initializer': 'he_normal',
    'skip_connection': False,
    'crop_center': True,

    'remove_dirty': 2,
    'radam': True,
})

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

        print('training loss:', self.loss(to_cs_st(self.train_set[1]), train_pred))
        print('validation loss:', self.loss(to_cs_st(self.valid_set[1]), valid_pred))

def train():
    event_names = ['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality']
    g = DataGenerator(remove_dirty=wandb.config.remove_dirty, event_names=event_names)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=False, classes=len(event_names))
    model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), loss=negative_hazard_log_likelihood)
    model.summary()
    wandb.log({'model_params': model.count_params()}, commit=False)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LossChecker(train_set, valid_set),
        ConcordanceIndex(train_set, valid_set, event_names),
        LogBest(records=['val_loss', 'loss'] + list(map(lambda x: '{}_cindex'.format(x),event_names)) + list(map(lambda x: 'val_{}_cindex'.format(x), event_names))),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    train_set = shuffle(train_set[0], train_set[1])
    valid_set = shuffle(valid_set[0], valid_set[1])
    model.fit(train_set[0], to_cs_st(train_set[1]), batch_size=64, epochs=500,
                validation_data=(valid_set[0], to_cs_st(valid_set[1])),
                callbacks=callbacks, shuffle=True)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))
    evaluation(model, test_set, event_names)

if __name__ == '__main__':
    train()
