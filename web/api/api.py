from flask import Flask, request
import numpy as np

import io
from flask import abort

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import shap

import base64
import keras
import tensorflow as tf
import keras.backend as K
import pickle
import json
import multiprocessing as mp

from visualization import plot_ekg

import time

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# import abnormal_detection.data_generator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'abnormal_detection'))
from data_generator import DataGenerator as ABDataGenerator

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.audicor_reader import denoise
from ekg.audicor_reader import reader
from ekg.layers import LeftCropLike, CenterCropLike
from ekg.layers.sincnet import SincConv1D

# initialization
app = Flask(__name__)
model_graph = tf.get_default_graph()
ABNORMALLY_MODEL_FILENAMES = ['../../abnormal_detection/models/2iw3s1f6.h5',
                                '../../abnormal_detection/models/2upywmpr.h5',
                                 '../../abnormal_detection/models/n6afydz2.h5']
ABNORMALLY_MODELS = [keras.models.load_model(fn, custom_objects={'SincConv1D': SincConv1D, 'LeftCropLike': LeftCropLike}) for fn in ABNORMALLY_MODEL_FILENAMES]
for model in ABNORMALLY_MODELS:
    model.predict(model.layers[0].inputs)
with open('../../abnormal_detection/models/means_and_stds.pl', 'rb') as f:
    ABNORMALLY_MEANS_AND_STDS = pickle.load(f)
ABNORMALLY_TRAINING_SET = ABDataGenerator(remove_dirty=2).get()[0]

def map2layer(x, layer):
    feed_dict = dict(zip([ABNORMALLY_MODELS[0].layers[0].input], [x.copy()]))
    return K.get_session().run(ABNORMALLY_MODELS[0].layers[layer].input, feed_dict)

with model_graph.as_default():
    layer_to_explain = 0
    ABNORMALLY_EXPLAINER = shap.GradientExplainer(
        (ABNORMALLY_MODELS[0].layers[layer_to_explain].input, ABNORMALLY_MODELS[0].layers[-1].output),
        map2layer(ABNORMALLY_TRAINING_SET[0], layer_to_explain),
        local_smoothing=0 # std dev of smoothing noise
    )

HAZARD_MODEL_FILENAMES = ['../../hazard_prediction/models/xtkcq47j.h5',
                            '../../hazard_prediction/models/b6pjvltd.h5',
                            '../../hazard_prediction/models/0o7ud2am.h5']
HAZARD_MODELS = [keras.models.load_model(fn, custom_objects={'SincConv1D': SincConv1D,
                                                        'LeftCropLike': LeftCropLike,
                                                        'CenterCropLike': CenterCropLike},
                                                        compile=False) for fn in HAZARD_MODEL_FILENAMES]
with open('../../hazard_prediction/models/means_and_stds.pl', 'rb') as f:
    HAZARD_MEANS_AND_STDS = pickle.load(f)

def get_ekg(f, do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    f.seek(0xE8)
    data_length = int.from_bytes(f.read(2), byteorder='little', signed=False)

    f.seek(0xE0)
    number_channels_ekg = int.from_bytes(f.read(2), byteorder='little', signed=False)

    f.seek(0xE4)
    number_channels_hs = int.from_bytes(f.read(2), byteorder='little', signed=False) # heart sound
    number_channels = number_channels_ekg + number_channels_hs

    data = [ list() for _ in range(number_channels) ]

    # data start
    f.seek(0x4B8)
    for index_cycle in range(data_length):
        raw = f.read(2 * number_channels)
        if len(raw) < 2 * number_channels:
            break
        for index_channel in range(number_channels):
            data[index_channel].append(int.from_bytes(
            raw[index_channel*2: (index_channel+1)*2],
            byteorder='little', signed=True))

    data = np.array(data)
    if do_bandpass_filter:
        for index_channel in range(number_channels_ekg, number_channels_ekg+number_channels_hs):
            data[index_channel] = reader.butter_bandpass_filter(data[index_channel], filter_lowcut, filter_highcut, 1000)
    return data, [1000.]*number_channels # sampling rates

def fig_encode(fig):
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue())

def normalize(X, means_and_stds):
    if means_and_stds is None:
        means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
        stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
    else:
        means = means_and_stds[0]
        stds = means_and_stds[1]

    normalized_X = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[-1]):
        normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
        normalized_X[..., i] = normalized_X[..., i] / stds[i]
    return normalized_X

def abnormally_preprocessing(ekg_signal):
    ekg_processed = ekg_signal[np.newaxis, ...].copy() # 1, 10000, 10
    ekg_processed = normalize(ekg_processed, ABNORMALLY_MEANS_AND_STDS)

    return ekg_processed

def hazard_preprocessing(ekg_signal):
    ekg_processed = ekg_signal[np.newaxis, ...].copy() # 1, 10000, 10
    ekg_processed = normalize(ekg_processed, HAZARD_MEANS_AND_STDS)

    return ekg_processed

def abnormally_score(ekg_signal):
    abnormally_score_start_time = time.time()

    # predict
    global model_graph
    with model_graph.as_default():
        prediction_scores = np.array([model.predict(ekg_signal)[0, 1] for model in ABNORMALLY_MODELS])

    print('prediction time:', time.time() - abnormally_score_start_time)

    return prediction_scores.mean()

def hazard_prediction(ekg_signal):
    start_time = time.time()

    # predict
    global model_graph
    with model_graph.as_default():
        prediction_scores = np.array([model.predict(ekg_signal)[0, ...] for model in HAZARD_MODELS])

    print('hazard prediction time:', time.time() - start_time)
    return prediction_scores.mean(axis=0)

def abnormally_explainer(normalized_ekg_signal):
    def smoothing(shap_value):
        # processing signals
        processed_singals = np.zeros_like(shap_value)
        for index_channel in range(shap_value.shape[-1]):
            if index_channel >= 8: # heart sound
                s = np.convolve(shap_value[..., index_channel], np.ones((200,))/200., mode='same')
            else: # ekg
                s = np.convolve(shap_value[..., index_channel], np.ones((50,))/50., mode='same')
            processed_singals[..., index_channel] = s

        return processed_singals

    start_time = time.time()
    shap_values = ABNORMALLY_EXPLAINER.shap_values(map2layer(normalized_ekg_signal, layer_to_explain), ranked_outputs=np.array([[1]]), output_rank_order='custom')

    shap_value = smoothing(shap_values[0][0][0]) # (10000, 10)

    shap_time = time.time()
    print('shap time:', shap_time - start_time)

    return shap_value

def plot_and_encode(signal, shap_value=None, figsize=None):
    return fig_encode(plot_ekg(signal, shap_value, figsize))

def predict(ekg_raw): # 10, 10000
    start_time = time.time()

    # denoise
    ekg_processed = ekg_raw.copy() # 10, 10000
    ekg_processed = denoise.denoise(ekg_processed, number_channels=8)
    ekg_processed = np.rollaxis(ekg_processed, 1, 0) # 10000, 10

    denoise_end_time = time.time()
    print('denoise time', denoise_end_time - start_time)

    # generate image in the background
    with mp.Pool(processes=2) as workers:
        denoised_ekg_plot = workers.apply_async(plot_and_encode, (ekg_processed, ))

        first_fig_time = time.time()
        print('first fig time', first_fig_time - denoise_end_time)

        # abnormally detection
        abnormally_signal = abnormally_preprocessing(ekg_processed)
        ab_shap_value = abnormally_explainer(abnormally_signal)

        ab_shap_plot = workers.apply_async(plot_and_encode, (ekg_processed, ab_shap_value))
        ab_score = abnormally_score(abnormally_signal)

        # hazard prediction
        hazard_signal = hazard_preprocessing(ekg_processed)
        hazard_score = hazard_prediction(hazard_signal)

        rtn_dict = {
            'ekg_plot': denoised_ekg_plot.get().decode(),
            'abnormally_score': ab_score.tolist(),
            'abnormally_explainer_plot': ab_shap_plot.get().decode(),
            'hazard_score': hazard_score.tolist()
        }

    return json.dumps(rtn_dict)

@app.before_request
def limit_remote_addr():
    client_ip = str(request.remote_addr)
    if not (client_ip.startswith('140.113.') or client_ip.startswith('127.0.0.1')): abort(404)

@app.route("/submit_ekg", methods=["POST"])
def receive_ekg():
    start_time = time.time()

    fileob = request.files["ekg_raw_file"]

    # decode ekg raw file
    ekg_raw, sampling_rates = get_ekg(fileob)
    result_json = predict(ekg_raw)
    print('final time:', time.time() - start_time)
    return result_json

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=9999)
