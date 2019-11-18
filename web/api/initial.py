import configparser
import numpy as np
import keras
import pickle
import shap

import keras.backend as K

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D

config = configparser.ConfigParser()
config.read('./configs.ini')

def load_models(task='abnormally'):
    model_filenames = config[task]['model_filenames'].split(',')
    models = [keras.models.load_model(fn.strip(),
                                custom_objects={'SincConv1D': SincConv1D,
                                                'LeftCropLike': LeftCropLike,
                                                'CenterCropLike': CenterCropLike},
                                compile=False)
                                for fn in model_filenames]

    with open(config['abnormally']['means_and_stds_filename'], 'rb') as f:
        means_and_stds = pickle.load(f)

    return models, means_and_stds

def make_random_predictions(graph, model):
    input_shape = [1] + list(model.layers[0].input_shape[1:])
    with graph.as_default():
        return model.predict(np.random.rand(*input_shape))

def load_shap_explainer(graph, training_set, model):
    with graph.as_default():
        # select a set of background examples to take an expectation over
        background = training_set[np.random.choice(training_set.shape[0], 500, replace=False)]

        # explain predictions of the model on four images
        explainer = shap.DeepExplainer(model, background)

    return explainer
