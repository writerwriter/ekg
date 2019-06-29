import keras
import keras.backend as K

class LeftCropLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        return source[:, source_shape[1]-target_shape[1]: ]

    def compute_output_shape(self, input_shape):
        return input_shape[1]