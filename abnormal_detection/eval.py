import os, sys
import numpy as np
import sklearn.metrics
import better_exceptions; better_exceptions.hook()

from data_generator import DataGenerator
from keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.layers import LeftCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.utils.eval_utils import print_cm

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def evaluation(model, test_set):
    print('Testing set baseline:', 1. - test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0])

    y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
    y_true = test_set[1][:, 1]

    print('Total accuracy:', sklearn.metrics.accuracy_score(y_true, y_pred))
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    model = load_model(sys.argv[1], custom_objects={'SincConv1D': SincConv1D, 'LeftCropLike': LeftCropLike}, compile=False)
    model.summary()
    train_set, valid_set, test_set = DataGenerator(remove_dirty=2).get()
    evaluation(model, test_set)
