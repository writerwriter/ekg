import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import wandb

def allow_gpu_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def set_wandb_config(params, overwrite=False):
    for key, value in params.items():
        if key not in wandb.config._items or overwrite:
            wandb.config.update({key: value})
