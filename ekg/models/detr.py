import pickle
import tensorflow as tf
import numpy as np
import wandb
from tensorflow.keras.layers import Conv1D, ReLU

from .backbone import backbone as branch_backbone
from .custom_layers import Linear
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from ..losses import AFTLoss


class DETR(tf.keras.Model):
    def __init__(self, num_classes=1, num_queries=3, backbone=None, pos_encoder=None, transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        
        self.backbone = backbone or branch_backbone(wandb.config, include_top=True, classification=False, classes=len(wandb.config.events))
        self.transformer = transformer or Transformer(return_intermediate_dec=True, name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(num_pos_features=self.model_dim, normalize=True)

        self.input_proj = Conv1D(self.model_dim, kernel_size=1, name='input_proj')
        self.query_embed = tf.Variable(tf.zeros((num_queries, self.model_dim), dtype=tf.float32), name='query_embed/kernel')

        self.class_embed = Linear(num_classes, name='class_embed')

        self.loss_layer = AFTLoss(len(wandb.config.events), 
                        wandb.config.AFT_distribution, 
                        wandb.config.AFT_initial_sigma,
                        name='AFT_loss')
    
    def call(self, inp, training=False, post_process=False):
        if 'cs_input' in inp:
            x = inp['ekg_hs_input']
            cs_input = inp['cs_input']
            st_input = inp['st_input']
            x = self.backbone(x)

            masks = tf.zeros_like(x, dtype=bool)[:,:,0]
            pos_encoding = self.pos_encoder(masks)
            #tf.print(pos_encoding)

            hs = self.transformer(self.input_proj(x), masks, self.query_embed, pos_encoding, training='cs_input' in inp)[0]
            
            output = self.class_embed(hs)[-1]

            output = self.loss_layer([cs_input, st_input, output])

        else:
            x = inp['ekg_hs_input']

            x = self.backbone(x)
            mask = np.ones_like(x, dtype=bool)
            pos_encoding = self.pos_encoder(masks)

            hs = self.transformer(self.input_proj(x), masks, self.query_embed, pos_encoding, training='cs_input' in inp)[0]

            output = self.class_embed(hs)[-1]

        return output

        """
    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = {'cs_input': (None, len(wandb.config.events)), 'st_input': (None, len(wandb.config.events)), 'ekg_hs_input': (None, None, 2)}
            #input_shape = [(None, None, 2), (None, None)]
        super().build(input_shape, **kwargs)
        """
    
    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        # fake image
        masks = tf.expand_dims(masks, -1)
        # channel
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(masks, [tf.shape(x)[1], 1], align_corners=False, half_pixel_centers=False)
        # remove channel
        masks = tf.squeeze(masks, -1)
        # remove fake image dim
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks,tf.bool)
        return masks

    def load_from_pickle(self, pickle_file, verbose=False):
        with open(pickle_file, 'rb') as f:
            detr_weights = pickle.load(f)
        
        for var in self.variables:
            if verbose:
                print('Loading', var.name)
            var = var.assign(detr_weights[var.name[:-2]])