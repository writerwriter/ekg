import numpy as np
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ekg.utils import data_utils

NUM_PROCESSES = mp.cpu_count()

def mp_generate_wavelet(hs, fs, desc):
    '''
    Args:
        hs: (?, n_channels, n_samples)
    '''
    n_instances = hs.shape[0]
    hs = hs.reshape(-1, hs.shape[-1])

    wavelets = list()
    __mp_gw = partial(data_utils.generate_wavelet, fs=fs)

    with mp.Pool(processes=NUM_PROCESSES) as workers:
        for wi in tqdm(workers.imap(__mp_gw, hs), total=hs.shape[0], desc=desc):
            wavelets.append(wi)

    wavelets = np.array(wavelets)
    return wavelets.reshape((n_instances, -1, wavelets.shape[-2], wavelets.shape[-1]))

def generate_wavelet(data_dir, hs_indices, fs):
    abnormal_X = np.load(os.path.join(data_dir, 'abnormal_X.npy')) # (?, n_channels, n_samples)
    normal_X = np.load(os.path.join(data_dir, 'normal_X.npy')) # (?, n_channels, n_samples)

    abnormal_hs = abnormal_X[:, hs_indices, :]
    normal_hs = normal_X[:, hs_indices, :]

    abnormal_wavelet    = mp_generate_wavelet(abnormal_hs, fs, 'generating abnormal wavelets')
    normal_wavelet      = mp_generate_wavelet(normal_hs, fs, 'generating normal wavelets')

    return abnormal_wavelet, normal_wavelet

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('./config.cfg')

    BIG_EXAM_OUTPUT_DIR = config['Big_Exam']['output_dir']
    AUDICOR_OUTPUT_DIR = config['Audicor_10s']['output_dir']

    abnormal_wavelet, normal_wavelet = generate_wavelet(BIG_EXAM_OUTPUT_DIR, [8, 9], 1000)
    np.save(os.path.join(BIG_EXAM_OUTPUT_DIR, 'abnormal_wavelet.npy'), abnormal_wavelet)
    np.save(os.path.join(BIG_EXAM_OUTPUT_DIR, 'normal_wavelet.npy'), normal_wavelet)

    abnormal_wavelet, normal_wavelet = generate_wavelet(AUDICOR_OUTPUT_DIR, [1], 500)
    np.save(os.path.join(AUDICOR_OUTPUT_DIR, 'abnormal_wavelet.npy'), abnormal_wavelet)
    np.save(os.path.join(AUDICOR_OUTPUT_DIR, 'normal_wavelet.npy'), normal_wavelet)
