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

def map2layer(model, x, layer):
    feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

def load_shap_explainer(graph, training_set, model):
    with graph.as_default():
        layer_to_explain = 0
        explainer = shap.GradientExplainer(
            (model.layers[layer_to_explain].input, model.layers[-1].output),
            map2layer(model, training_set, layer_to_explain),
            local_smoothing=0 # std dev of smoothing noise
        )
    return explainer
