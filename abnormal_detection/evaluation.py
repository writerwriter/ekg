import os, sys
import numpy as np
import sklearn.metrics
import argparse
import better_exceptions; better_exceptions.hook()

from keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()

from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D
from ekg.utils.eval_utils import print_cm

def evaluation(models, test_set):
    print('Testing set baseline:', 1. - test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0])

    y_pred_vote = np.zeros_like(test_set[1][:, 1])
    try: # model ensemble
        for model in models:
            y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
            y_pred_vote = y_pred_vote + y_pred

        y_pred = (y_pred_vote > (len(models) / 2)).astype(int)
    except:
        y_pred = np.argmax(models.predict(test_set[0], batch_size=64), axis=1)

    y_true = test_set[1][:, 1]

    print('Total accuracy:', sklearn.metrics.accuracy_score(y_true, y_pred))
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    # NOTE: THIS IS NOT WORKING RIGHT NOW.
    from train import DataGenerator
    import configparser
    from ekg.utils.eval_utils import YamlParser

    parser = argparse.ArgumentParser(description='Abnormal detection evaluation.')
    parser.add_argument('-ne', dest='n_ekg_channels', type=int, default=8,
                        help='Number of ekg channels.')
    parser.add_argument('-nh', dest='n_hs_channels', type=int, default=2,
                        help='Number of heart sound channels.')
    parser.add_argument('wandb_yaml_config', metavar='wandb_yaml_config', type=str,
                        help='Path to the wandb yaml configuration file.')
    parser.add_argument('model_filenames', metavar='filename', type=str, nargs='+',
                        help='Filenames of models to be evaluated.')

    args = parser.parse_args()

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike
    }

    models = list()
    for fn in args.model_filenames:
        models.append(load_model(fn, custom_objects=custom_objects, compile=False))

    models[0].summary()

    # read config
    config = configparser.ConfigParser()
    config.read('./config.cfg')

    # parse wandb config yaml file
    wandb_config = YamlParser().read(args.wandb_yaml_config)
    
    train_set, valid_set, test_set = DataGenerator(config, wandb_config).get()
    evaluation(models, test_set)
