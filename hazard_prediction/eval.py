import os, sys
import numpy as np
import argparse
import better_exceptions; better_exceptions.hook()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ekg.utils import train_utils; train_utils.allow_gpu_growth()

from data_generator import DataGenerator
from keras.models import load_model

from ekg.layers import LeftCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.layers import CenterCropLike

from lifelines.utils import concordance_index

def ensemble_predict(models, X, batch_size=64):
    y_preds = list()
    for i, m in enumerate(models):
        y_preds.append(m.predict(X, batch_size=batch_size))

    y_pred = np.stack(y_preds, axis=2).mean(axis=-1) # (?, n_events, num_models) -> (?, n_events)
    return y_pred

def evaluation(models, test_set, event_names):
    '''
        test_set[1]: (num_samples, num_events, 2)
    '''
    X, y = test_set[0], test_set[1]

    try:
        y_pred = ensemble_predict(models, X)

    except Exception:
        y_pred = models.predict(X, batch_size=64)

    # calculate cindex
    for i in range(len(event_names)): # for every event
        event_cs, event_st = y[:, i, 0], y[:, i, 1]
        cindex = concordance_index(event_st, -y_pred[:, i], event_cs) # cindex of the event
        print('Concordance index of {} testing set: {:.4f}'.format(event_names[i], cindex))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hazard prediction evaluation.')
    parser.add_argument('model_filenames', metavar='filename', type=str, nargs='+',
                        help='filenames of models to be evaluated.')

    args = parser.parse_args()

    event_names=['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality']
    models = list()
    for fn in args.model_filenames:
        models.append(load_model(fn, custom_objects={'SincConv1D': SincConv1D,
                                                        'LeftCropLike': LeftCropLike,
                                                        'CenterCropLike': CenterCropLike}, compile=False))

    models[0].summary()
    train_set, valid_set, test_set = DataGenerator(remove_dirty=2, event_names=event_names).get()
    evaluation(models, test_set, event_names)
