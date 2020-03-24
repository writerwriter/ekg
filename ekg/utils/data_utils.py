import numpy as np
import pandas as pd
import itertools
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

import better_exceptions; better_exceptions.hook()

from ..audicor_reader import reader

NUM_PROCESSES = mp.cpu_count()*2

class DataAugmenter:
    def __init__(self, indices_channel_ekg, indices_channel_hs,
                    ekg_scaling_prob, hs_scaling_prob,
                    time_stretch_prob):

        self.indices_channel_ekg = indices_channel_ekg
        self.indices_channel_hs = indices_channel_hs
        self.total_nchannel = len(self.indices_channel_hs) + len(self.indices_channel_ekg)

        self.ekg_scaling_prob = ekg_scaling_prob
        self.hs_scaling_prob = hs_scaling_prob
        self.time_stretch_prob = time_stretch_prob

    # input shape: (10000, 10)
    def ekg_scaling(self, X, ratio):
        X[..., self.indices_channel_ekg] *= ratio

    def hs_scaling(self, X, ratio):
        X[..., self.indices_channel_hs] *= ratio

    def random_crop(self, X, length):
        original_length = X.shape[-2]
        random_offset = np.random.choice(np.arange(0, original_length-length+1, 1))
        return X[random_offset: random_offset+length, :]

    def time_stretch(self, X, ratio):
        rtn = list()
        original_length = X.shape[-2]
        for i in range(X.shape[-1]): # through every channel
            rtn.append(np.interp(np.linspace(0, original_length, ratio*original_length+1), np.arange(original_length), X[..., i]))

        rtn = np.array(rtn).swapaxes(0, 1) # (10000, 10)
        return self.random_crop(rtn, original_length)

    def augment(self, X):
        X  = X.copy().astype(np.float32)

        if np.random.rand() <= self.ekg_scaling_prob:
            self.ekg_scaling(X, np.random.uniform(0.95, 1.05))

        if np.random.rand() <= self.hs_scaling_prob:
            self.hs_scaling(X, np.random.uniform(0.95, 1.05))

        if np.random.rand() <= self.time_stretch_prob:
            X = self.time_stretch(X, np.random.uniform(1.0, 1.05))

        return X

class BaseDataGenerator:
    def __init__(self, big_exam_dir, audicor_10s_dir,
                    wandb_config, **kwargs):

        self.big_exam_dir = big_exam_dir
        self.audicor_10s_dir = audicor_10s_dir
        self.config = wandb_config

        self.big_exam_channel_set = self.calculate_channel_set(self.config.big_exam_ekg_channels,
                                                                    self.config.big_exam_hs_channels)
        self.audicor_10s_channel_set = self.calculate_channel_set(self.config.audicor_10s_ekg_channels,
                                                                    self.config.audicor_10s_hs_channels)
        
        self.big_exam_n_repeat = len(self.big_exam_channel_set)
        self.audicor_10s_n_repeat = len(self.audicor_10s_channel_set)

        self.big_exam_n_instances = 0
        self.audicor_10s_n_instances = 0

        # set additional variables
        self.__dict__.update(kwargs)

        # X shape: (n_instances, n_samples, n_channels)
        self.abnormal_X = self.get_abnormal_X() # abnormal first
        self.normal_X = self.get_normal_X()

        # subject_id shape: (n_instances)
        self.abnormal_subject_id = self.get_abnormal_subject_id() # abnormal first
        self.normal_subject_id = self.get_normal_subject_id()

        # y shape: (n_instances, ...)
        self.abnormal_y = self.get_abnormal_y() # abnormal first
        self.normal_y = self.get_normal_y()

        self.means_and_stds = None # [means, stds], the shapes of means and stds are both [n_channels].

        self.preprocessing()

    def calculate_channel_set(self, ekg_channels, hs_channels):
        def to_list(x):
            return [list(xi) for xi in x]
        def flatten(x):
            return [xi[0] + xi[1] for xi in x]
        all_ekg_channel_set = to_list(itertools.combinations(ekg_channels, self.config.n_ekg_channels))
        all_hs_channel_set = to_list(itertools.combinations(hs_channels, self.config.n_hs_channels))

        return flatten(to_list(itertools.product(all_ekg_channel_set, all_hs_channel_set)))

    def load_X(self, filename):
        '''
        Outputs:
            X: np.ndarray of shape [n_instances, n_samples, n_channels]
        '''
        X = np.zeros((0, self.config.n_ekg_channels+self.config.n_hs_channels, self.config.sampling_rate*10))
        
        if 'big_exam' in self.config.datasets:
            big_exam_X = np.load(os.path.join(self.big_exam_dir, filename)) # (n_instances, n_channels, n_samples)
            if self.config.sampling_rate != 1000:
                big_exam_X = downsample(big_exam_X, 
                                        1000 // self.config.sampling_rate,
                                        self.config.downsample,
                                        channels_last=False)
            # select channels
            for channel_set in self.big_exam_channel_set:
                X = np.append(X, big_exam_X[:, channel_set, :], axis=0)

            self.big_exam_n_instances = big_exam_X.shape[0] * len(self.big_exam_channel_set)

        if 'audicor_10s' in self.config.datasets:
            audicor_10s_X = np.load(os.path.join(self.audicor_10s_dir, filename)) # (n_instances, n_channels, n_samples)

            # select channels
            for channel_set in self.audicor_10s_channel_set:
                X = np.append(X, audicor_10s_X[:, channel_set, :], axis=0)

            self.audicor_10s_n_instances = audicor_10s_X.shape[0] * len(self.audicor_10s_channel_set)

        # make X channel last
        return np.swapaxes(X, 1, 2)

    def load_subject_id(self, is_normal):
        # abnormal
        subject_id = np.empty((0, ), dtype=str)
        if 'big_exam' in self.config.datasets:
            if is_normal: # assume normal data are all from different subjects
                big_exam_subject_id = np.arange(self.big_exam_n_instances // self.big_exam_n_repeat, dtype=int)

            else:
                # read abnormal_event for suject id
                df = pd.read_csv(os.path.join(self.big_exam_dir, 'abnormal_event.csv'))
                big_exam_subject_id = df.subject_id.values

            for _ in self.big_exam_channel_set:
                subject_id = np.append(subject_id, big_exam_subject_id, axis=0)

        if 'audicor_10s' in self.config.datasets:
            if is_normal: # assume normal data are all from different subjects
                audicor_10s_subject_id = np.load(os.path.join(self.audicor_10s_dir, 'normal_filenames.npy'))

            else:
                audicor_10s_subject_id = np.load(os.path.join(self.audicor_10s_dir, 'abnormal_filenames.npy'))
                # get the filenames by spliting by '/'
                audicor_10s_subject_id = np.vectorize(lambda fn: fn.split('/')[-1])(audicor_10s_subject_id)

                # get the subject ids by spliting by '_'
                audicor_10s_subject_id = np.vectorize(lambda fn: fn.split('_')[0])(audicor_10s_subject_id)

            for _ in self.audicor_10s_channel_set:
                subject_id = np.append(subject_id, audicor_10s_subject_id, axis=0)

        return subject_id

    def get_abnormal_X(self):
        return self.load_X('abnormal_X.npy')
    
    def get_abnormal_y(self):
        '''return abnormal_y

        The shape must be (n_instances, ...)
        '''
        return NotImplementedError('get_abnormal_y not implemented!')

    def get_abnormal_subject_id(self):
        return self.load_subject_id(is_normal=False)

    def get_normal_X(self):
        return self.load_X('normal_X.npy')

    def get_normal_y(self):
        '''return normal_y

        The shape must be (n_instances, ...)
        '''
        return NotImplementedError('get_normal_y not implemented!')

    def get_normal_subject_id(self):
        return self.load_subject_id(is_normal=True)

    def preprocessing(self):
        '''Do whatever you want LOL.
        '''
        NotImplementedError('preprocessing not implemented!')

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

    def get_split(self, rs=42):
        '''Split the data into training set, validation set, and testing set.

        Outputs:
            train_set, valid_set, test_set
        '''
        def combine(set1, set2):
            return [np.append(set1[i], set2[i], axis=0) for i in range(2)]

        # do abnormal split by abnormal subject ID
        abnormal_training_set, abnormal_valid_set, abnormal_test_set  = subject_split(self.abnormal_X, self.abnormal_y, self.abnormal_subject_id, rs)

        # do normal split by normal subject ID
        normal_training_set, normal_valid_set, normal_test_set = subject_split(self.normal_X, self.normal_y, self.normal_subject_id, rs)

        # combine
        train_set = combine(normal_training_set, abnormal_training_set)
        valid_set = combine(normal_valid_set, abnormal_valid_set)
        test_set = combine(normal_test_set, abnormal_test_set)

        return [train_set, valid_set, test_set]

    def get(self):
        '''Split the data into training set, validation set, and testing set, and then normalize them by training set.

        Outputs:
            train_set, valid_set, test_set
        '''
        train_set, valid_set, test_set = self.get_split()

        # do normalize using means and stds from training data
        train_set[0], self.means_and_stds = self.normalize(train_set[0])
        valid_set[0], _ = self.normalize(valid_set[0], self.means_and_stds)
        test_set[0], _ = self.normalize(test_set[0], self.means_and_stds)

        return train_set, valid_set, test_set

def to_spectrogram(data):
    rtn = list()

    __mp_generate_spectrogram = partial(reader.generate_spectrogram, sampling_rates=[1000.]*data.shape[-2]) # ?, 10, 10000
    with mp.Pool(processes=32) as workers:
        for result in tqdm(workers.imap(__mp_generate_spectrogram, data), total=data.shape[0], desc='To spectrogram'):
            rtn.append([Sxx for f, t, Sxx in result])

    return np.array(rtn)

def patient_split(*args, **kwargs):
    import warnings
    warnings.warn('The patient_split() method is deprecated, use subject_split() instead!', UserWarning)
    return subject_split(*args, **kwargs)

def subject_split(X, y, subject_id, rs=42):
    '''Split the X and y by the subject ID by the ratio (n_train: n_valid: n_test = 0.49: 0.21: 0.3).

    Args:
        X: np.ndarray of shape [n_instance, ...]
        y: np.ndarray of shape [n_instance, ...]

    Returns:
        [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
    '''
    unique_subject_id = np.unique(np.array(subject_id))

    # split subject_id
    train_id, test_id = train_test_split(unique_subject_id, test_size=0.3, random_state=rs)
    train_id, valid_id = train_test_split(train_id, test_size=0.3, random_state=rs)

    _m_train    = [pi in train_id for pi in subject_id]
    _m_test     = [pi in test_id for pi in subject_id]
    _m_valid    = [pi in valid_id for pi in subject_id]

    X_train, y_train    = X[_m_train],  y[_m_train]
    X_test, y_test      = X[_m_test],   y[_m_test]
    X_valid, y_valid    = X[_m_valid],  y[_m_valid]

    return [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]

def downsample(X, ratio=2, mode='direct', channels_last=False):
    '''Downsample by either direct mode or averaging mode.

    Args:
        X: shape [n_instances, n_samples, n_channels] if channels_last else
                    [n_instances, n_channels, n_samples]
        ratio: downsample ratio, integer only for now
        mode: either direct or average

    Outputs:
        Downsampled_X
    '''
    if mode == 'direct':
        if channels_last:
            return X[..., ::ratio, :]
        else:
            return X[..., ::ratio]
    elif mode == 'average':
        if channels_last:
            return (X[..., ::ratio, :] + X[..., 1::ratio, :]) / 2.0
        else:
            return (X[..., ::ratio] + X[..., 1::ratio]) / 2.0
    else:
        raise ValueError('Invalid mode')

def calculate_n_ekg_channels(config):
    n_ekg_channels = 99999
    if 'big_exam' in config.datasets:
        n_ekg_channels = min(n_ekg_channels, len(config.big_exam_ekg_channels))
    
    if 'audicor_10s' in config.datasets:
        n_ekg_channels = min(n_ekg_channels, len(config.audicor_10s_ekg_channels))

    return n_ekg_channels

def calculate_n_hs_channels(config):
    n_hs_channels = 99999
    if 'big_exam' in config.datasets:
        n_hs_channels = min(n_hs_channels, len(config.big_exam_hs_channels))
    
    if 'audicor_10s' in config.datasets:
        n_hs_channels = min(n_hs_channels, len(config.audicor_10s_hs_channels))

    return n_hs_channels

if __name__ == '__main__':
    pass
