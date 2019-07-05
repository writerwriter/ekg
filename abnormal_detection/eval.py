import os, sys
import numpy as np
import sklearn.metrics
import argparse
import better_exceptions; better_exceptions.hook()

# allow_growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from data_generator import DataGenerator
from keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.layers import LeftCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.utils.eval_utils import print_cm

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def evaluation(models, test_set):
    print('Testing set baseline:', 1. - test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0])

    y_pred_vote = np.zeros_like(test_set[1][:, 1])
    for model in models:
        y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
        y_pred_vote = y_pred_vote + y_pred

    y_pred = (y_pred_vote > (len(models) / 2)).astype(int)
    y_true = test_set[1][:, 1]

    print('Total accuracy:', sklearn.metrics.accuracy_score(y_true, y_pred))
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Abnormal detection evaluation.')
    parser.add_argument('model_filenames', metavar='filename', type=str, nargs='+',
                        help='filenames of models to be evaluated.')

    args = parser.parse_args()

    models = list()
    for fn in args.model_filenames:
        models.append(load_model(fn, custom_objects={'SincConv1D': SincConv1D, 'LeftCropLike': LeftCropLike}, compile=False))

    models[0].summary()
    train_set, valid_set, test_set = DataGenerator(remove_dirty=2).get()
    evaluation(models, test_set)
