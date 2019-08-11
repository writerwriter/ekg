import pickle
import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback
wandb.init(project='ekg-segmentation', entity='toosyou')

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import Callback
import os

from ekg.callbacks import LogBest
from model import unet_1d
from data_generator import DataGenerator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

set_wandb_config({
    'target': '24hr',
    'seg_setting': 'pqrst',

    'amsgrad': True,
    'n_middle_lstm': 3,
    'ending_lstm': False,
    # 'model_padding': 'valid',

    # data
    'sample_weight_moveing_average': True,
    'window_moving_average': 6,
})


class PredictPlotter(Callback):
    def __init__(self, plot_sample):
        '''
            plot_sample: assumably normalized
        '''
        self.plot_sample = plot_sample
        # fix the sample shape, which should be 3 (1, signal_length, n_channels)
        if len(self.plot_sample.shape) == 2:
            self.plot_sample = self.plot_sample[np.newaxis, ...]
        super().__init__()

    def _generate_plot(self, pred):
        def _to_array(fig):
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return img

        fig, axes = plt.subplots(ncols=1, nrows=6, sharex=True, figsize=(20, 10))

        for i, ax in enumerate(axes):
            ax.set_title('pqrst-'[i])
            ax.plot(pred[0, :, i], color='C0')

            ax = ax.twinx()
            ax.plot(self.plot_sample[0, :, 0], color='C1')

        plt.margins(x=0.01, y=0.01)
        return _to_array(fig)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            pred = self.model.predict(self.plot_sample)
            img = self._generate_plot(pred)
            wandb.log({'predict': wandb.Image(img)}, commit=False)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

def get_weight(y, do_moving_average=False, window_size=10):
    weight_matrix = np.max(np.swapaxes(y, 1, 2)[:, :5, :], axis=1)*1000 + 1.       # background weight 1 , else 1001
    if do_moving_average:
        for i in range(weight_matrix.shape[0]):
            weight_matrix[i] = np.pad(moving_average(weight_matrix[i], n=window_size),
                                        (window_size//2, window_size//2-1), mode='constant', constant_values=1)
    return weight_matrix

def train():
    g = DataGenerator(wandb.config)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    training_weight = get_weight(train_set[1],
                                wandb.config.sample_weight_moveing_average,
                                wandb.config.window_moving_average)
    validation_weight = get_weight(valid_set[1],
                                wandb.config.sample_weight_moveing_average,
                                wandb.config.window_moving_average)

    model = unet_1d(wandb.config)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        PredictPlotter(plot_sample=valid_set[0][0]),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=2000, validation_data=(valid_set[0], valid_set[1], validation_weight), callbacks=callbacks, shuffle=True, sample_weight=training_weight)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

if __name__ == '__main__':
    train()
