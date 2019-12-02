import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import re
import pandas as pd
import datetime
import warnings

import better_exceptions; better_exceptions.hook()

from ekg.utils.data_utils import load_patient_data, load_normal_data
from ekg.audicor_reader import denoise

def preprocessing(X):
    '''
        X: (?, 10, 10000)
    '''
    warnings.filterwarnings("ignore")
    for i, xi in enumerate(tqdm(X, desc='preprocessing')):
        X[i] = denoise.denoise(xi, number_channels=8) # only denoise the first 8 ekg channels

    return

def generate_survival_data(event_names=['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality']):
    def get_info(path_string):
        path_parts = path_string.split('/')

        patient_code = path_parts[1]
        patient_measuring_date = None
        if re.match(r'(^(A|a|H|c|V|[0-9]))', path_parts[2]):
            patient_measuring_date = re.findall(r"[0-9]{3,}", path_parts[2])[0]

        return patient_code, patient_measuring_date

    path_df = pd.read_csv('./data/LVEF_path.csv', header=None, names=['path', 'LVEF']).drop(columns=['LVEF'])
    ahf2017_df = pd.read_excel('./data/AHF2017_outcome_anonymous.xls', skiprows=1)

    follow_durs = list()
    durs = dict()
    for event_name in event_names: durs[event_name] = list()

    for index, path in path_df.itertuples():
        code, measuring_date = get_info(path)
        ahf2017_df_row = ahf2017_df[ahf2017_df.code == code]

        if measuring_date is None:
            follow_dur = int(ahf2017_df_row.follow_dur)
        else:
            follow_dur = int(((ahf2017_df_row.follow_date) - datetime.datetime.strptime(measuring_date, '%m%d%y')).dt.days)
        follow_durs.append(follow_dur)

        for event_name in event_names:
            try:
                if measuring_date is None: # don't get measuring date in path, find it in outcome.csv
                    dur = int(ahf2017_df_row[event_name + '_dur'])
                else:
                    dur = int(((ahf2017_df_row[event_name + '_date']) - datetime.datetime.strptime(measuring_date, '%m%d%y')).dt.days)
            except:
                dur = np.nan
            durs[event_name].append(dur)

    path_df['follow_dur'] = follow_durs
    for event_name, dur in durs.items():
        path_df[event_name + '_dur'] = dur

    return path_df

def generate_ekg_data():
    normal_X = load_normal_data()
    preprocessing(normal_X)
    np.save('data/normal_X.npy', normal_X)

    patient_X, patient_id = load_patient_data(remove_dirty=0)
    preprocessing(patient_X)
    np.save('data/patient_X.npy', patient_X)
    np.save('data/patient_id.npy', patient_id)

    cleaned_patient_X, cleaned_patient_id = load_patient_data(remove_dirty=1)
    preprocessing(cleaned_patient_X)
    np.save('data/cleaned_1_patient_X.npy', cleaned_patient_X)
    np.save('data/cleaned_1_patient_id.npy', cleaned_patient_id)

    cleaned_patient_X, cleaned_patient_id = load_patient_data(remove_dirty=2)
    preprocessing(cleaned_patient_X)
    np.save('data/cleaned_2_patient_X.npy', cleaned_patient_X)
    np.save('data/cleaned_2_patient_id.npy', cleaned_patient_id)
    return None

if __name__ == '__main__':
    generate_ekg_data()
    generate_survival_data().to_csv('./data/path_dur.csv', index=False)
