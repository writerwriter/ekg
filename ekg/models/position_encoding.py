import numpy as np
import tensorflow as tf
import sys

class PositionEmbeddingSine(tf.keras.Model):
    def __init__(self, num_pos_features=64, temperature=10000, normalize=False, scale=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps

    def call(self, mask):
        not_mask = tf.cast(~mask, tf.float32)
        embed = tf.math.cumsum(not_mask, axis=1)

        #tf.print(embed)

        if self.normalize:
            embed = embed / (embed[:, -1:] + self.eps) * self.scale
        
        dim_t = tf.range(self.num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        pos = embed[..., tf.newaxis] / dim_t

        pos = tf.stack([tf.math.sin(pos[..., 0::2]), tf.math.cos(pos[..., 1::2])], axis=3)

        shape = [tf.shape(pos)[i] for i in range(2)] + [-1]
        pos = tf.reshape(pos, shape)

        return pos