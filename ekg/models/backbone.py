import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate
# from keras.layers import Maximum

from ..layers import LeftCropLike
from ..layers.sincnet import SincConv1D
from ..layers.non_local import non_local_block

def backbone(config, include_top=False, classification=True, classes=2): # TODO: add residual connection
    def heart_sound_branch(hs):
        sincconv_filter_length = config.sincconv_filter_length - (config.sincconv_filter_length+1) % 2
        hs = SincConv1D(config.sincconv_nfilters, sincconv_filter_length, 1000)(hs)
        hs = BatchNormalization()(hs)

        for _ in range(config.branch_nlayers):
            hs = Conv1D(8, config.hs_kernel_length, activation='relu', padding='same')(hs)
            hs = BatchNormalization()(hs)
            hs = MaxPooling1D(3, padding='same')(hs)
        return hs

    total_input = Input((None, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    # ekg branch
    ekg = ekg_input
    for _ in range(config.branch_nlayers):
        ekg = Conv1D(8, config.ekg_kernel_length, activation='relu', padding='same')(ekg_input)
        ekg = BatchNormalization()(ekg)
        ekg = MaxPooling1D(3, padding='same')(ekg)

    # heart sound branch
    hs_outputs = list()
    hs = Lambda(lambda x: K.expand_dims(x[:, :, 0], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 1], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Add()(hs_outputs)
    ekg = LeftCropLike()([ekg, hs])
    output = Concatenate(axis=-1)([hs, ekg])

    if include_top: # final layers
        for i in range(config.final_nlayers):
            output = Conv1D(8, config.final_kernel_length, activation='relu', padding='same')(output)
            output = BatchNormalization()(output)

            if i >= config.final_nlayers - config.final_nonlocal_nlayers: # the final 'final_nonlocal_nlayers' layers
                output = output = non_local_block(output, compression=2, mode='embedded')

            if i != config.final_nlayers-1: # not the final output
                output = MaxPooling1D(2, padding='same')(output)

        output = GlobalAveragePooling1D()(output)
        if classification:
            output = Dense(classes, activation='softmax')(output)
        else: # regression
            output = Dense(1, activation='linear')(output)

    model = Model(total_input, output)
    return model
