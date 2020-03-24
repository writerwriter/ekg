import pickle
import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback
wandb.init(project='ekg-segmentation')

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import Callback
import os

from ekg.callbacks import LogBest
from model import unet_lstm
from data_generator import DataGenerator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import configs
# set_wandb_config(configs.original_unet)

set_wandb_config({
    'target': '24hr',
    'seg_setting': 'pqrst',

    'amsgrad': True,
    'radam': True,
    'n_encoding_layers': 8,
    'n_initial_layers': 2,
    'n_conv_per_encoding_layer': 3,
    'kernel_size_encoding': 7,
    'index_middle_lstm': 5,
    'n_middle_lstm': 3,
    'n_middle_lstm_units': 8,
    'n_final_conv': 6,
    'base_feature_number': 8,
    'max_feature_number': 128,
    'ending_lstm': False,
    'model_padding': 'same',
    'bidirectional': True,
    'batch_normalization': False,

    # data
    'peak_weight': 0,
    'window_moving_average': 0,
    'window_weight_forgiveness': 0,

    'regression': False,
    'label_blur_kernel': 11,
    'label_normalization': True,
    'label_normalization_value': 512, # TODO: calculate it by peak and background ratio
    'use_all_leads': True,
})



class PredictPlotter(Callback):
    def __init__(self, plot_sample, y_weight=(None,None), config=wandb.config, model_output_shape=[None, 5000, 6], freq=50):
        '''
            plot_sample: assumably normalized
            y_weight: (y, weight)
                        y: [signal_length, number_channels]
                        weight: [signal_length]

        '''
        self.plot_sample = plot_sample
        self.config = config
        self.model_output_shape = model_output_shape
        self.freq = freq
        self.y = y_weight[0]
        self.weight = y_weight[1]
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

        if self.config.model_padding == 'valid':
            diff = 5000 - self.model_output_shape[1]
            original_signal = self.plot_sample[0, diff//2: (diff//2 + self.model_output_shape[1]) ,0]
        else:
            original_signal = self.plot_sample[0, :, 0]

        number_plot_channel = pred.shape[2] + 1 if self.y is not None else pred.shape[2]
        fig, axes = plt.subplots(ncols=1, nrows=number_plot_channel, sharex=True, figsize=(20, 10))
        for i, ax in enumerate(axes):
            if i == len(axes) - 1: # the last channel is for ground truth
                ax.set_title('ground truth')
                for index_channel in range(self.y.shape[1]):
                    ax.plot(self.y[:, index_channel], color='C0')
                ax.plot(-self.weight / self.weight.max() * self.y.max(), color='C2')

                ax = ax.twinx()
                ax.plot(original_signal, color='C1')
            else:
                ax.set_title('pqrst-'[i])
                ax.plot(pred[0, :, i], color='C0')

                ax = ax.twinx()
                ax.plot(original_signal, color='C1')

        plt.margins(x=0.01, y=0.01)
        return _to_array(fig)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            pred = self.model.predict(self.plot_sample)
            img = self._generate_plot(pred)
            wandb.log({'predict': wandb.Image(img)}, commit=False)

def get_weight(y, peak_weight = 1000, window_moving_average=10, window_forgiveness=0):
    '''
        y.shape == [?, signal_length, n_channels]
    '''
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:]

    peak_matrix = y[:, :, :5].max(axis=2) # (?, signal_length)

    weight_matrix = peak_matrix * peak_weight + 1.         # background weight 1 , else 1001, # (?, signal_length)
    if window_moving_average > 0:
        for i in range(weight_matrix.shape[0]):
            weight_matrix[i] = np.pad(moving_average(weight_matrix[i], n=window_moving_average),
                                        (window_moving_average//2, window_moving_average//2), mode='constant', constant_values=1)

    if window_forgiveness > 0:
        for i in range(weight_matrix.shape[0]):
            forgiveness_matrix = np.pad(moving_average(peak_matrix[i], n=window_forgiveness),
                                        (window_forgiveness//2, window_forgiveness//2), mode='constant', constant_values=0) - peak_matrix[i]
            weight_matrix[i, forgiveness_matrix==1] = 0

    return weight_matrix

def train():
    model, model_output_shape = unet_lstm(wandb.config)
    model.summary()
    wandb.log({'model_ouptut_length': model_output_shape[1]}, commit=False)
    wandb.log({'model_params': model.count_params()}, commit=False)

    g = DataGenerator(wandb.config, model_output_shape)
    train_set, valid_set, test_set = g.get()

    # save means and stds to wandb
    with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        pickle.dump(g.means_and_stds, f)

    training_weight = get_weight(train_set[1],
                                wandb.config.peak_weight,
                                wandb.config.window_moving_average,
                                wandb.config.window_weight_forgiveness)
    validation_weight = get_weight(valid_set[1],
                                wandb.config.peak_weight,
                                wandb.config.window_moving_average,
                                wandb.config.window_weight_forgiveness)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        PredictPlotter(plot_sample=valid_set[0][0], 
                        y_weight=(valid_set[1][0], validation_weight[0]), 
                        config=wandb.config, model_output_shape=model_output_shape, freq=20),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64,
                    epochs=6000, validation_data=(valid_set[0], valid_set[1], validation_weight),
                    callbacks=callbacks, shuffle=True, sample_weight=training_weight)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

if __name__ == '__main__':
    train()
