import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import data

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class DataGenerator:
    def __init__(self, config, model_output_shape=None):
        """
        self.patient_X = np.load(os.path.join(DATA_DIR, 'seg_X.npy')) # (90, 8, 10000)
        self.patient_y= np.load(os.path.join(DATA_DIR, 'seg_y.npy')) # (90, 6, 10000)
        """

        self.means_and_stds = None
        self.config = config
        self.model_output_shape = model_output_shape

        self.X, self.y = None, None
        self.preprocessing()

    def preprocessing(self):
        xy = data.load_dataset()
        self.X = xy['x']
        self.y = xy['y']

        """
        self.X = np.swapaxes(self.patient_X, 1, 2) # (90, 10000, 8)
        self.y = np.swapaxes(self.patient_y, 1, 2) # (90, 10000, 6) # pqrst

        if self.config.target == '24hr':
            # downsample the signals to 500hz, while the original ones are 1000Hz
            self.X = self.X[:, ::2]

            if not self.config.use_all_leads:
                self.X = self.X[:, :, 0:1] # and only use the first lead signals

            self.y = self.y[:, ::2, :] + self.y[:, 1::2, :]

            if self.config.model_padding == 'valid':
                diff = 5000 - self.model_output_shape[1]
                self.y = self.y[:, diff//2: (diff//2 + self.model_output_shape[1]) ,:]

        else: # bigexam
            pass
        """
        if self.config.regression:
            # remove the last channel of y
            self.y = self.y[:, :, :-1]

            # apply gaussian filter to label
            ksize = self.config.label_blur_kernel
            sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
            for index_sample in range(self.y.shape[0]):
                for index_channel in range(self.y.shape[2]):
                    self.y[index_sample, :, index_channel] = gaussian_filter(self.y[index_sample, :, index_channel], sigma=sigma)

            # normalization
            if self.config.label_normalization:
                for index_sample in range(self.y.shape[0]):
                    for index_channel in range(self.y.shape[2]):
                        self.y[index_sample, :, index_channel] /= self.y[index_sample, :, index_channel].max() / self.config.label_normalization_value

        if self.config.seg_setting == 'split':
            print('ERROR: seg_setting {} not supported!'.format(self.config.seg_setting))

    @staticmethod
    def normalize(X, means_and_stds=None):
        if means_and_stds is None:
            means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
            stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
        else:
            means = means_and_stds[0]
            stds = means_and_stds[1]

        normalized_X = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[-1]):
            normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
            normalized_X[..., i] = normalized_X[..., i] / stds[i]
        return normalized_X, (means, stds)

    def get(self):
        def __signal_flatten(dataset):
            X, y = dataset

            # X: (?, signal_length, 8) -> (?, 8, signal_length) -> (? * 8, signal_length, 1)
            X = np.moveaxis(X, -1, 1)
            X = np.reshape(X, (-1, X.shape[-1]))
            X = X[..., np.newaxis]

            # y: (?, signal_length, 5 or 6) -> (? * 8, signal_length, 5 or 6)
            y = np.repeat(y, 12, axis=0)

            return [X, y]

        train_set, valid_set, test_set = self.split()

        if self.config.target == '24hr' and self.config.use_all_leads: # flatten all leads
            # (?, signal_length, 8) -> (? * 8, signal_length, 1)
            train_set = __signal_flatten(train_set)
            valid_set = __signal_flatten(valid_set)
            test_set = __signal_flatten(test_set)

        # do normalize using means and stds from training data
        train_set[0], self.means_and_stds = DataGenerator.normalize(train_set[0])
        valid_set[0], _ = DataGenerator.normalize(valid_set[0], self.means_and_stds)
        test_set[0], _ = DataGenerator.normalize(test_set[0], self.means_and_stds)

        return train_set, valid_set, test_set

    def split(self, rs=42):
        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=rs)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=rs)

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
