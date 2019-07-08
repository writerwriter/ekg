import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate
# from keras.layers import Maximum

from ..layers import LeftCropLike
from ..layers.sincnet import SincConv1D
from ..layers.non_local import non_local_block

def _ekg_block(input, nlayers, kernel_length, skip_connection):
    ekg = input
    for i in range(nlayers):
        shortcut = ekg
        ekg = Conv1D(8, kernel_length, activation='relu', padding='same', name='ekg_branch_conv{}'.format(i))(ekg)
        ekg = BatchNormalization(name='ekg_branch_bn{}'.format(i))(ekg)

        if skip_connection:
            ekg = Add(name='ekg_branch_skip_merge{}'.format(i))([ekg, shortcut])

        ekg = MaxPooling1D(3, padding='same', name='ekg_branch_maxpool{}'.format(i))(ekg)

    return ekg

def backbone(config, include_top=False, classification=True, classes=2): # TODO: add residual connection
    def heart_sound_branch(hs, name_prefix=''):
        sincconv_filter_length = config.sincconv_filter_length - (config.sincconv_filter_length+1) % 2
        hs = SincConv1D(config.sincconv_nfilters, sincconv_filter_length, 1000, name='{}sincconv'.format(name_prefix))(hs)
        hs = BatchNormalization(name='{}bn0'.format(name_prefix))(hs)

        for i in range(config.branch_nlayers):
            hs = Conv1D(8, config.hs_kernel_length, activation='relu', padding='same', name='{}conv_{}'.format(name_prefix, i+1))(hs)
            hs = BatchNormalization(name='{}bn{}'.format(name_prefix, i+1))(hs)
            hs = MaxPooling1D(3, padding='same', name='{}maxpool_{}'.format(name_prefix, i+1))(hs)
        return hs

    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8], name='ekg_input')(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:], name='hs_input')(total_input) # (10000, 2)

    # ekg branch
    ekg = ekg_input
    for i in range(config.branch_nlayers):
        ekg = Conv1D(8, config.ekg_kernel_length, activation='relu', padding='same', name='ekg_branch_conv_{}'.format(i))(ekg)
        ekg = BatchNormalization(name='ekg_branch_bn_{}'.format(i))(ekg)
        ekg = MaxPooling1D(3, padding='same', name='ekg_branch_maxpool_{}'.format(i))(ekg)

    # heart sound branch
    hs_outputs = list()
    hs = Lambda(lambda x: K.expand_dims(x[:, :, 0], -1), name='hs_split_0')(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs, name_prefix='hs_branch_0_'))

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 1], -1), name='hs_split_1')(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs, name_prefix='hs_branch_1_'))

    hs = Add(name='hs_merge')(hs_outputs)
    ekg = LeftCropLike(name='ekg_crop')([ekg, hs])
    output = Concatenate(axis=-1, name='hs_ekg_merge')([hs, ekg])

    if include_top: # final layers
        for i in range(config.final_nlayers):
            output = Conv1D(8, config.final_kernel_length, activation='relu', padding='same', name='final_conv_{}'.format(i))(output)
            output = BatchNormalization(name='final_bn_{}'.format(i))(output)

            if i >= config.final_nlayers - config.final_nonlocal_nlayers: # the final 'final_nonlocal_nlayers' layers
                output = output = non_local_block(output, compression=2, mode='embedded')

            if i != config.final_nlayers-1: # not the final output
                output = MaxPooling1D(2, padding='same', name='final_maxpool_{}'.format(i))(output)

        output = GlobalAveragePooling1D()(output)
        if classification:
            output = Dense(classes, activation='softmax')(output)
        else: # regression
            output = Dense(1, activation='linear')(output)

    model = Model(total_input, output)
    return model
