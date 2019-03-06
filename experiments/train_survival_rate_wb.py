import numpy as np
import better_exceptions; better_exceptions.hook()
import keras
import sklearn.metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from models import get_survival_rate_wb_model
import utils
from utils import patient_split

from datetime import datetime
import pickle

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class DataGenerator:
    def __init__(self):
        '''
        valid_blood_labels = ['BUNV2', 'CrV2', 'eGFR2', 'SodiumV2',
                                'potassiumV2', 'HgbV2', 'Alb', 'UA', 'Tbil',
                                'proBNP', 'PBNP', 'TNT', 'adiponectin',
                                'Troponin_T', 'nST2', 'pGal3', 'hsCRP',
                                'Aldosterone', 'nPRA', 'nAdiponectin',
                                'mmp9', 'TIMP1', 'LVEF']
        '''
        valid_blood_labels = ['BUNV2', 'CrV2', 'eGFR2', 'SodiumV2', 'potassiumV2',
                                'HgbV2', 'Alb', 'UA', 'Tbil', 'proBNP', 'PBNP', 'TNT', 'adiponectin',
                                'Troponin_T', 'nST2', 'pGal3', 'hsCRP', 'Aldosterone', 'nPRA',
                                'nAdiponectin', 'mmp9', 'TIMP1', 'LVOTmm', 'AoRmm', 'LAmm',
                                'IVSdmm', 'LVIDdmm', 'PWdmm', 'LVIDsmm', 'MVEcms', 'MVAcms',
                                'MVEdtmsec', 'EAdurmsec', 'RR2msec', 'IVRTmsec', 'PVS2cms',
                                'PVDcms', 'PVDdtmsec', 'PVAcms', 'AoVTIcm', 'v_4chEDVml',
                                'v_4chESVml', 'bpEDVml', 'bpESVml', 'RSacms', 'REacms', 'RAacms',
                                'LSSacms', 'LSEacms', 'LSAacms', 'LLSacms', 'LLEacms', 'LLAacms',
                                'OuterDdmm', 'InnerDdmm', 'InnerDsmm', 'IMTmm', 'LVM', 'BSAV2',
                                'LVMI', 'med_Eea', 'mean_Eea', 'EA_ratio', 'RWT', 'EEa']

        self.patient_X = np.load('./patient_X.npy') # (852, 10, 10000)
        self.blood = np.ndarray((self.patient_X.shape[0], len(valid_blood_labels)), dtype=float) # (852, 22)
        for i, bl in enumerate(valid_blood_labels):
            self.blood[:, i] = utils.load_target(bl, dtype=float)

        self.patient_event = utils.load_target('MACE')
        self.patient_event_dur = utils.load_target('MACE_dur') # (852)
        self.preprocessing()

    def preprocessing(self):
        def dur2target(dur):
            # 1 year, 2 years, 3 years, 4 years and above
            y = np.zeros((dur.shape[0], 4), dtype=np.float)
            for i in range(4):
                # y[:, i] = np.clip(dur - 365*i, 0., 365.) / 365.
                y[:, i] = (dur >= 365*(i+1)) | (self.patient_event == 0) # nothing happened = [1, 1, 1, 1]
            return y

        def padding_avg(blood):
            masked_blood = np.ma.masked_equal(np.ma.masked_invalid(blood), 0)
            means = masked_blood.mean(axis=0)
            for i in range(masked_blood.shape[1]):
                masked_blood[:, i] = masked_blood[:, i].filled(means[i])

            return masked_blood

        # combine normal and patient
        # self.X = np.append(self.patient_X, self.normal_X, axis=0) # (?, 10, 10000)
        self.X_ekg = self.patient_X
        self.X_blood = padding_avg(self.blood)
        # self.y = np.append(dur2target(self.patient_event_dur), np.ones((self.normal_X.shape[0], 4)), axis=0) # (?, 4)
        self.y = dur2target(self.patient_event_dur)

        # change dimension
        # ?, n_channels, n_points -> ?, n_points, n_channels
        self.X_ekg = np.swapaxes(self.X_ekg, 1, 2)

        self.X = [self.X_ekg, self.X_blood]

        # print('baseline:', self.y.sum() / self.y.shape[0])
        self.print_baseline(self.y)

    def X_shape(self):
        return self.X.shape[1:]

    @staticmethod
    def print_baseline(y):
        for i in range(4):
            print('baseline - ({:d}, {:d}]: {:.2f}'.format(i*365, (i+1)*365, y[:, i].sum() / y.shape[0]) )

    @staticmethod
    def normalize(X, means_and_stds=None):
        X_ekg, X_blood = X # (852, 10000, 10), (852, 22)

        if means_and_stds is None:
            ekg_means = [ X_ekg[..., i].mean(dtype=np.float32) for i in range(X_ekg.shape[-1]) ]
            blood_means = [ X_blood[..., i].mean(dtype=np.float32) for i in range(X_blood.shape[-1]) ]

            ekg_stds = [ X_ekg[..., i].std(dtype=np.float32) for i in range(X_ekg.shape[-1]) ]
            blood_stds = [ X_blood[..., i].std(dtype=np.float32) for i in range(X_blood.shape[-1]) ]
        else:
            ekg_means, ekg_stds, blood_means, blood_stds = means_and_stds

        normalized_X_ekg = X_ekg.copy()
        for i in range(X_ekg.shape[-1]):
            normalized_X_ekg[..., i] = X_ekg[..., i].astype(np.float32) - ekg_means[i]
            normalized_X_ekg[..., i] = normalized_X_ekg[..., i] / ekg_stds[i]

        normalized_X_blood = X_blood.copy()
        for i in range(X_blood.shape[-1]):
            normalized_X_blood[..., i] = X_blood[..., i].astype(np.float32) - blood_means[i]
            normalized_X_blood[..., i] = normalized_X_blood[..., i] / blood_stds[i]

        return [normalized_X_ekg, normalized_X_blood], [ekg_means, ekg_stds, blood_means, blood_stds]

    def data(self):
        return self.X, self.y

    @staticmethod
    def split(X, y, rs=42):
        X_ekg, X_blood = X

        ekg_training_set, ekg_valid_set, ekg_test_set = patient_split(X_ekg, y, rs)
        blood_training_set, blood_valid_set, blood_test_set = patient_split(X_blood, y, rs)

        X_train = [ekg_training_set[0], blood_training_set[0]]
        y_train = ekg_training_set[1]

        X_valid = [ekg_valid_set[0], blood_valid_set[0]]
        y_valid = ekg_valid_set[1]

        X_test = [ekg_test_set[0], blood_test_set[0]]
        y_test = ekg_test_set[1]

        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]

def train():
    g = DataGenerator()
    X, y = g.data()
    train_set, valid_set, test_set = g.split(X, y)

    model_checkpoints_dirname = 'sr_wb_model_checkpoints/'+datetime.now().strftime('%Y%m%d_%H%M_%S')
    tensorboard_log_dirname = model_checkpoints_dirname + '/logs'
    os.makedirs(model_checkpoints_dirname)
    os.makedirs(tensorboard_log_dirname)

    # do normalize using means and stds from training data
    train_set[0], means_and_stds = DataGenerator.normalize(train_set[0])
    valid_set[0], _ = DataGenerator.normalize(valid_set[0], means_and_stds)
    test_set[0], _ = DataGenerator.normalize(test_set[0], means_and_stds)

    # save means and stds
    with open(model_checkpoints_dirname + '/means_and_stds.pl', 'wb') as f:
        pickle.dump(means_and_stds, f)

    model = get_survival_rate_wb_model()
    model.summary()

    callbacks = [
        # EarlyStopping(patience=5),
        ModelCheckpoint(model_checkpoints_dirname + '/{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        TensorBoard(log_dir=tensorboard_log_dirname)
    ]

    print(train_set[0][0].shape, train_set[0][1].shape)

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=500, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True, class_weight=[1/0.58, 1/0.4, 1/0.22, 1/0.1])

    # y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
    # y_true = test_set[1][:, 1]
    # print(sklearn.metrics.classification_report(y_true, y_pred))

    # print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    train()
