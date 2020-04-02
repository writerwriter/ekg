import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate
# from keras.layers import Maximum

from ..layers import LeftCropLike, CenterCropLike
from ..layers.sincnet import SincConv1D
from ..layers.non_local import non_local_block

def _ekg_branch(input_data, nlayers, kernel_length, kernel_initializer, skip_connection):
    ekg = input_data
    for i in range(nlayers):
        shortcut = ekg
        ekg = Conv1D(8, kernel_length, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='ekg_branch_conv_{}'.format(i))(ekg)
        ekg = BatchNormalization(name='ekg_branch_bn_{}'.format(i))(ekg)

        if skip_connection:
            ekg = Add(name='ekg_branch_skip_merge_{}'.format(i))([ekg, shortcut])

        ekg = MaxPooling1D(3, padding='same', name='ekg_branch_maxpool_{}'.format(i))(ekg)

    return ekg

def _heart_sound_branch(input_data, sincconv_filter_length, sincconv_nfilters, sampling_rate, nlayers, kernel_length, kernel_initializer, skip_connection, name_prefix=''):
    hs = input_data
    sincconv_filter_length = sincconv_filter_length - (sincconv_filter_length+1) % 2
    hs = SincConv1D(sincconv_nfilters, sincconv_filter_length, sampling_rate, name='{}sincconv'.format(name_prefix))(hs)
    hs = BatchNormalization(name='{}bn_0'.format(name_prefix))(hs)

    for i in range(nlayers):
        shortcut = hs
        hs = Conv1D(8, kernel_length, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='{}conv_{}'.format(name_prefix, i+1))(hs)
        hs = BatchNormalization(name='{}bn_{}'.format(name_prefix, i+1))(hs)

        if skip_connection:
            shortcut = Conv1D(8, 1, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='{}skip_bottleneck_{}'.format(name_prefix, i+1))(shortcut)
            hs = Add(name='{}skip_merge_{}'.format(name_prefix, i+1))([hs, shortcut])

        hs = MaxPooling1D(3, padding='same', name='{}maxpool_{}'.format(name_prefix, i+1))(hs)

    return hs

def backbone(config, include_top=False, classification=True, classes=2):
    total_input = Input((config.sampling_rate*10, config.n_ekg_channels + config.n_hs_channels))
    
    # ekg branch
    if config.n_ekg_channels != 0:
        ekg_input = Lambda(lambda x, n_ekg_channels: x[:, :, :n_ekg_channels], 
                                    arguments={'n_ekg_channels': config.n_ekg_channels}, 
                                    name='ekg_input')(total_input) # (10000, 8)
        ekg = _ekg_branch(ekg_input, config.branch_nlayers, config.ekg_kernel_length, config.kernel_initializer, config.skip_connection)

    # heart sound branch
    if config.n_hs_channels != 0:
        heart_sound_input = Lambda(lambda x, n_hs_channels: x[:, :, -n_hs_channels:], 
                                    arguments={'n_hs_channels': config.n_hs_channels}, 
                                    name='hs_input')(total_input) # (10000, 2)

        hs_outputs = list()
        for i in range(config.n_hs_channels):
            hs = Lambda(lambda x, i: K.expand_dims(x[:, :, i], -1), 
                                    arguments={'i': i}, 
                                    name='hs_split_{}'.format(i))(heart_sound_input)
            hs_outputs.append(_heart_sound_branch(hs, config.sincconv_filter_length,
                                                        config.sincconv_nfilters, 
                                                        config.sampling_rate,
                                                        config.branch_nlayers,
                                                        config.hs_kernel_length, config.kernel_initializer,
                                                        config.skip_connection, name_prefix='hs_branch_{}_'.format(i)))
        if config.n_hs_channels >= 2:
            hs = Add(name='hs_merge')(hs_outputs)
        else: # no need to merge
            hs = hs_outputs[0]

    # merge block
    if config.n_ekg_channels != 0 and config.n_hs_channels != 0:
        if config.crop_center:
            ekg = CenterCropLike(name='ekg_crop')([ekg, hs])
        else:
            ekg = LeftCropLike(name='ekg_crop')([ekg, hs])
        output = Concatenate(axis=-1, name='hs_ekg_merge')([hs, ekg])
    else:
        output = ekg if config.n_ekg_channels != 0 else hs

    if include_top: # final layers
        for i in range(config.final_nlayers):
            shortcut = output
            output = Conv1D(8, config.final_kernel_length, activation='relu', padding='same',
                                kernel_initializer=config.kernel_initializer, name='final_conv_{}'.format(i))(output)
            output = BatchNormalization(name='final_bn_{}'.format(i))(output)

            if config.skip_connection:
                if i == 0:
                    shortcut = Conv1D(8, 1, activation='linear', padding='same',
                                        kernel_initializer=config.kernel_initializer, name='final_shortcut_conv')(shortcut)
                output = Add(name='final_skip_merge_{}'.format(i))([output, shortcut])

            if i >= config.final_nlayers - config.final_nonlocal_nlayers: # the final 'final_nonlocal_nlayers' layers
                output = non_local_block(output, compression=2, mode='embedded')

            if i != config.final_nlayers-1: # not the final output
                output = MaxPooling1D(2, padding='same', name='final_maxpool_{}'.format(i))(output)

        output = GlobalAveragePooling1D()(output)
        output = Dense(classes, activation='softmax' if classification else 'linear')(output) # classification or regression

    model = Model(total_input, output)
    return model
