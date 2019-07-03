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
DIRTYDATA_INDICES_FILENAME = config['Paths']['dirtydata_indices']

BIG_EXAM_DIR = config['Paths']['big_exam_dir']
BIG_EXAM_NORMAL_DIRS = config['Paths']['big_exam_normal_dirs'].split(', ')

NUM_PROCESSES = mp.cpu_count()*2

class DataAugmenter:
    def __init__(self, indices_channel_ekg, indices_channel_hs):
        self.indices_channel_ekg = indices_channel_ekg
        self.indices_channel_hs = indices_channel_hs

    def ekg_scaling(self, X, ratio):
        X[self.indices_channel_ekg] *= ratio

    def hs_scaling(self, X, ratio):
        X[self.indices_channel_hs] *= ratio

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

def load_patient_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100, remove_dirty=0):
    '''
        remove_dirty:
            0 - don't remove dirty data
            1 - remove dirty data
            2 - remove dirty data and not-so-dirty data
    '''
    def is_dirty(pi, dirtydata_indices_df):
        for dp in dirtydata_indices_df['dirty']:
            if dp == dp and dp in pi: return True
        if remove_dirty == 2: # not-so-dirty data
            for dp in dirtydata_indices_df['gray_zone']:
                if dp == dp and dp in pi: return True
        return False

    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])
    paths = path_lvef_df['path'].values
    # clean up
    if remove_dirty > 0:
        dirtydata_indices_df = pd.read_excel(DIRTYDATA_INDICES_FILENAME)
        paths = np.array([p for p in paths if not is_dirty(p, dirtydata_indices_df)])

    filenames = list(map(lambda x: BIG_EXAM_DIR+x, paths))
    patient_id = np.array([p.split('/')[1] for p in paths])
    X = mp_get_ekg(filenames, do_bandpass_filter, filter_lowcut, filter_highcut, 'Loading patient data')

    return X, patient_id

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
    patient_X, patient_id = load_patient_data(remove_dirty=2)
    # normal_X = load_normal_data()
