import numpy as np
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import better_exceptions; better_exceptions.hook()
import os

import sys
sys.path.append('../audicor_reader')
import reader
import denoise

def to_spectrogram(data):
    rtn = list()

    __mp_generate_spectrogram = partial(reader.generate_spectrogram, sampling_rates=[1000.]*data.shape[-2]) # ?, 10, 10000
    with mp.Pool(processes=32) as workers:
        for result in tqdm(workers.imap(__mp_generate_spectrogram, data), total=data.shape[0], desc='To spectrogram'):
            rtn.append([Sxx for f, t, Sxx in result])

    return np.array(rtn)

def load_normal_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    def get_normal_filenames():
        PREFIX = '/home/toosyou/ext_ssd/Cardiology/交大-normal/大檢查audicor'
        filenames = list()
        for directory, sub_directory, filelist in os.walk(PREFIX, topdown=True):
            for fn in filelist:
                if fn.endswith('.bin'):
                    filenames.append(os.path.join(directory, fn))
        return filenames

    filenames = get_normal_filenames()

    X = list()
    __mp_get_ekg = partial(reader.get_ekg, do_bandpass_filter=do_bandpass_filter, filter_lowcut=filter_lowcut, filter_highcut=filter_highcut)
    with mp.Pool(processes=16) as workers:
        for xi, _ in tqdm(workers.imap(__mp_get_ekg, filenames), total=len(filenames), desc='Load data'):
            X.append(xi)

    return np.array(X)

def load_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    path_lvef_df = pd.read_csv('./LVEF_path.csv', header=None, names=['path', 'LVEF'])

    PREFIX_BIN = '/home/toosyou/ext_ssd/Cardiology/交大數據心衰遠距教學/大檢查audicor'

    X, y = list(), list()
    pathes = map(lambda x: PREFIX_BIN+x, path_lvef_df['path'])
    __mp_get_ekg = partial(reader.get_ekg, do_bandpass_filter=do_bandpass_filter, filter_lowcut=filter_lowcut, filter_highcut=filter_highcut)
    with mp.Pool(processes=16) as workers:
        for (xi, _), lvef in tqdm(zip(workers.imap(__mp_get_ekg, pathes), path_lvef_df['LVEF']), total=path_lvef_df['path'].shape[0], desc='Load data'):
            X.append(xi)
            y.append(lvef)

    X, y = np.array(X), np.array(y)
    return X, y

def load_target(target_name):
    ahf2017_df = pd.read_excel('./AHF2017_outcome_anonymous.xls', skiprows=1)
    path_lvef_df = pd.read_csv('./LVEF_path.csv', header=None, names=['path', 'LVEF'])

    # get patient code
    patient_id = list()
    for p in path_lvef_df['path']:
        patient_id.append(p.split('/')[1])

    target = list()
    for p in patient_id:
        try:
            target.append(int(ahf2017_df[target_name][ahf2017_df.code == p]))
        except:
            target.append(0)

    return np.array(target)

def patient_split(X, y, rs=42):
    path_lvef_df = pd.read_csv('./LVEF_path.csv', header=None, names=['path', 'LVEF'])

    # get the unique patient ids
    patient_id = list()
    for p in path_lvef_df['path']:
        patient_id.append(p.split('/')[1])
    patient_id = np.unique(np.array(patient_id))

    # split patient id
    train_id, test_id = train_test_split(patient_id, test_size=0.3, random_state=rs)
    train_id, valid_id = train_test_split(train_id, test_size=0.3, random_state=rs)

    X_train, X_val, X_test = list(), list(), list()
    y_train, y_val, y_test = list(), list(), list()

    for i, p in enumerate(path_lvef_df['path']):
        pid = p.split('/')[1]
        if pid in train_id:
            X_train.append(X[i])
            y_train.append(y[i])
        elif pid in valid_id:
            X_val.append(X[i])
            y_val.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    return [[X_train, y_train], [X_val, y_val], [X_test, y_test]]

if __name__ == '__main__':
    patient_X, patient_y = load_data()
    normal_X = load_normal_data()

    # preprocessing
    __mp_denoise = partial(denoise.denoise, number_channels=8)
    with mp.Pool(processes=32) as workers:
        for i, xi in tqdm(enumerate(workers.imap(__mp_denoise, patient_X)), desc='preprocessing', total=patient_X.shape[0]):
            patient_X[i] = xi

    # preprocessing
    __mp_denoise = partial(denoise.denoise, number_channels=8)
    with mp.Pool(processes=32) as workers:
        for i, xi in tqdm(enumerate(workers.imap(__mp_denoise, normal_X)), desc='preprocessing', total=normal_X.shape[0]):
            normal_X[i] = xi

    np.save('patient_X.npy', patient_X)
    np.save('patient_y.npy', patient_y)
    np.save('normal_X.npy', normal_X)
