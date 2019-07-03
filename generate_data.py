import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import better_exceptions; better_exceptions.hook()

from ekg.utils.data_utils import load_patient_data, load_normal_data
from ekg.audicor_reader import denoise

def preprocessing(X):
    __mp_denoise = partial(denoise.denoise, number_channels=8)
    with mp.Pool(processes=mp.cpu_count()*2) as workers:
        for i, xi in tqdm(enumerate(workers.imap(__mp_denoise, X)), desc='preprocessing', total=X.shape[0]):
            X[i] = xi

if __name__ == '__main__':
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
