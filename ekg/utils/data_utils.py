import numpy as np
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

def to_spectrogram(data):
    rtn = list()

    __mp_generate_spectrogram = partial(reader.generate_spectrogram, sampling_rates=[1000.]*data.shape[-2]) # ?, 10, 10000
    with mp.Pool(processes=32) as workers:
        for result in tqdm(workers.imap(__mp_generate_spectrogram, data), total=data.shape[0], desc='To spectrogram'):
            rtn.append([Sxx for f, t, Sxx in result])

    return np.array(rtn)

def patient_split(X, y, patient_id, rs=42):
    unique_patient_id = np.unique(np.array(patient_id))

    # split patient id
    train_id, test_id = train_test_split(unique_patient_id, test_size=0.3, random_state=rs)
    train_id, valid_id = train_test_split(train_id, test_size=0.3, random_state=rs)

    _m_train    = [pi in train_id for pi in patient_id]
    _m_test     = [pi in test_id for pi in patient_id]
    _m_valid    = [pi in valid_id for pi in patient_id]

    X_train, y_train    = X[_m_train],  y[_m_train]
    X_test, y_test      = X[_m_test],   y[_m_test]
    X_valid, y_valid    = X[_m_valid],  y[_m_valid]

    return [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]

if __name__ == '__main__':
    pass
