import numpy as np
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import better_exceptions; better_exceptions.hook()
import os
import configparser

from ..audicor_reader import reader

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'configs.ini'))
OUTCOME_FILENAME = config['Paths']['outcome']
LVEF_PATH_FILENAME = config['Paths']['lvef_path']
LVEF_PATH_DUR_FILENAME = config['Paths']['lvef_path_dur']

BIG_EXAM_DIR = config['Paths']['big_exam_dir']
BIG_EXAM_NORMAL_DIRS = config['Paths']['big_exam_normal_dirs'].split(', ')

NUM_PROCESSES = mp.cpu_count()*2

def to_spectrogram(data):
    rtn = list()

    __mp_generate_spectrogram = partial(reader.generate_spectrogram, sampling_rates=[1000.]*data.shape[-2]) # ?, 10, 10000
    with mp.Pool(processes=32) as workers:
        for result in tqdm(workers.imap(__mp_generate_spectrogram, data), total=data.shape[0], desc='To spectrogram'):
            rtn.append([Sxx for f, t, Sxx in result])

    return np.array(rtn)

def mp_get_ekg(filenames, do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100, desc='Loading data'):
    X = list()
    __mp_get_ekg = partial(reader.get_ekg, do_bandpass_filter=do_bandpass_filter, filter_lowcut=filter_lowcut, filter_highcut=filter_highcut)
    with mp.Pool(processes=NUM_PROCESSES) as workers:
        for xi, _ in tqdm(workers.imap(__mp_get_ekg, filenames), total=len(filenames), desc=desc):
            X.append(xi)

    return np.array(X)

def load_normal_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    def get_normal_filenames(dirname):
        filenames = list()
        for directory, sub_directory, filelist in os.walk(dirname, topdown=True):
            for fn in filelist:
                if fn.endswith('.bin'):
                    filenames.append(os.path.join(directory, fn))
        return filenames

    filenames = list()
    for dirname in BIG_EXAM_NORMAL_DIRS:
        filenames = filenames + get_normal_filenames(dirname)
    return mp_get_ekg(filenames, do_bandpass_filter, filter_lowcut, filter_highcut, 'Loading normal data')

def load_patient_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])

    filenames = list(map(lambda x: BIG_EXAM_DIR+x, path_lvef_df['path']))
    X = mp_get_ekg(filenames, do_bandpass_filter, filter_lowcut, filter_highcut, 'Loading patient data')
    return X

def load_patient_LVEF():
    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])
    return path_lvef_df['LVEF'].values

def load_target(target_name, dtype=int):
    ahf2017_df = pd.read_excel(OUTCOME_FILENAME, skiprows=1)
    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])

    # get patient code
    patient_id = list()
    for p in path_lvef_df['path']:
        patient_id.append(p.split('/')[1])

    target = list()
    for p in patient_id:
        try:
            target.append(dtype(ahf2017_df[target_name][ahf2017_df.code == p]))
        except:
            target.append(0)

    return np.array(target)

def patient_split(X, y, rs=42):
    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])

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

def load_survival_data():
    dur_df = pd.read_csv(LVEF_PATH_DUR_FILENAME)

    # 1: event occurred, 0: survived, -1: pass event, ignore
    censoring_stats = np.ones(dur_df.path.shape) * -1
    survival_times = np.zeros(dur_df.path.shape)

    # dur_df.follow_dur[i] < 0: this measurement is done after the final follow date
    # dur_df.ADHF_dur[i] != dur_df.ADHF_dur[i]: no ADHF occurred, survived
    # dur_df.ADHF_dur[i] < 0: this measurement is done after event

    for i in range(censoring_stats.shape[0]):
        if dur_df.follow_dur[i] < 0 or\
            dur_df.ADHF_dur[i] < 0 or\
            'NG' in dur_df.path[i]: # the measurement is not trust-worthy
            censoring_stats[i] = -1

        elif dur_df.ADHF_dur[i] != dur_df.ADHF_dur[i]: # Nan
            censoring_stats[i] = 0 # survived
            survival_times[i] = dur_df.follow_dur[i]

        else: # event occurred
            censoring_stats[i] = 1
            survival_times[i] = dur_df.ADHF_dur[i]

    return censoring_stats, survival_times

if __name__ == '__main__':
    # patient_X = load_patient_data()
    normal_X = load_normal_data()
