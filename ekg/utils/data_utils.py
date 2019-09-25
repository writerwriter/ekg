import numpy as np
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import better_exceptions; better_exceptions.hook()
import os
import configparser
from functools import partial

from ..audicor_reader import reader

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'configs.ini'))
OUTCOME_FILENAME = config['Paths']['outcome']
LVEF_PATH_FILENAME = config['Paths']['lvef_path']
PATH_DUR_FILENAME = config['Paths']['path_dur']
DIRTYDATA_INDICES_FILENAME = config['Paths']['dirtydata_indices']

BIG_EXAM_DIR = config['Paths']['big_exam_dir']
BIG_EXAM_NORMAL_DIRS = config['Paths']['big_exam_normal_dirs'].split(', ')

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

def path_is_dirty(pi, dirtydata_indices_df, remove_dirty):
    for dp in dirtydata_indices_df['dirty']:
        if dp == dp and dp in pi: return True
    if remove_dirty == 2: # not-so-dirty data
        for dp in dirtydata_indices_df['gray_zone']:
            if dp == dp and dp in pi: return True
    return False

def load_patient_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100, remove_dirty=0):
    '''
        remove_dirty:
            0 - don't remove dirty data
            1 - remove dirty data
            2 - remove dirty data and not-so-dirty data
    '''
    path_lvef_df = pd.read_csv(LVEF_PATH_FILENAME, header=None, names=['path', 'LVEF'])
    paths = path_lvef_df['path'].values
    # clean up
    if remove_dirty > 0:
        dirtydata_indices_df = pd.read_excel(DIRTYDATA_INDICES_FILENAME)
        paths = np.array([p for p in paths if not path_is_dirty(p, dirtydata_indices_df, remove_dirty)])

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

def load_survival_data(event_name, remove_dirty=0):
    '''
        remove_dirty:
            0 - don't remove dirty data
            1 - remove dirty data
            2 - remove dirty data and not-so-dirty data
    '''
    dur_df = pd.read_csv(PATH_DUR_FILENAME)

    if remove_dirty > 0:
        dirtydata_indices_df = pd.read_excel(DIRTYDATA_INDICES_FILENAME)
        filter_function = lambda row: not path_is_dirty(row['path'], dirtydata_indices_df, remove_dirty)
        dur_df = dur_df[dur_df.apply(filter_function, axis=1)].reset_index(drop=True)

    # dur_df.follow_dur[i] < 0: this measurement is done after the final follow date
    # dur_df.ADHF_dur[i] != dur_df.ADHF_dur[i]: no ADHF occurred, survived
    # dur_df.ADHF_dur[i] < 0: this measurement is done after event

    event_dur_name = event_name + '_dur'

    # 1: event occurred, 0: survived, -1: pass event, ignore
    dur_df['censoring_stats'] = 1 # event occurred
    dur_df.loc[dur_df[event_dur_name] != dur_df[event_dur_name], 'censoring_stats'] = 0 # Nan
    dur_df.loc[(dur_df.follow_dur <= 0) | (dur_df[event_dur_name] < 0) | (dur_df.path.str.contains('NG')), 'censoring_stats'] = -1

    dur_df['survival_times'] = dur_df[event_dur_name] # event occurred
    dur_df.loc[dur_df[event_dur_name] != dur_df[event_dur_name], 'survival_times'] = dur_df.follow_dur[dur_df[event_dur_name] != dur_df[event_dur_name]] # Nan
    dur_df.loc[(dur_df.follow_dur <= 0) | (dur_df[event_dur_name] < 0) | (dur_df.path.str.contains('NG')), 'survival_times'] = 0

    censoring_stats = dur_df['censoring_stats'].values
    survival_times = dur_df['survival_times'].values.astype(int)

    return censoring_stats, survival_times

if __name__ == '__main__':
    print(load_survival_data('ADHF', 2))
    # patient_X, patient_id = load_patient_data(remove_dirty=2)
    # normal_X = load_normal_data()
