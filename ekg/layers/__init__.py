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

class CenterCropLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        length_diff = source_shape[1]-target_shape[1]
        return source[:, length_diff//2 : length_diff//2+target_shape[1] ]

    def compute_output_shape(self, input_shape):
        return input_shape[1]
