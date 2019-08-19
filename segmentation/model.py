from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, concatenate, UpSampling1D, Softmax, ZeroPadding1D, Bidirectional, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.layers import CenterCropLike

def unet_lstm(config):
    # NOTE: for now, it only work with config.model_padding == valid
    input_size = (5000, 1) if config.target == '24hr' else (10000, 8) # 'bigexam'

    input = Input(input_size)
    base_feature_number = config.base_feature_number

    net = input

    # encoding
    encoding_layers = list()
    for i in range(config.n_encoding_layers): # 5
        number_feature = min(base_feature_number*(2**i), config.max_feature_number)
        net = Conv1D(number_feature, 7, activation = 'relu',
                        padding = config.model_padding,
                        kernel_initializer = 'he_normal',
                        name='encoding_conv_{}'.format(i*2))(net)
        net = BatchNormalization(name='encoding_bn_{}'.format(i*2))(net) if config.batch_normalization else net
        net = Conv1D(number_feature, 7, activation = 'relu',
                        padding = config.model_padding,
                        kernel_initializer = 'he_normal',
                        name='encoding_conv_{}'.format(i*2+1))(net)
        net = BatchNormalization(name='encoding_bn_{}'.format(i*2+1))(net) if config.batch_normalization else net
        encoding_layers.append(net)

        if i != config.n_encoding_layers - 1:
            net = MaxPooling1D(pool_size=2, name='encoding_maxpool_{}'.format(i))(net)

    # middle lstms
    for _ in range(config.n_middle_lstm):
        lstm_layer = Bidirectional(LSTM(units=32, return_sequences=True)) if config.bidirectional else LSTM(units=32, return_sequences=True)
        net = lstm_layer(net)

    # decoding
    for i in range(config.n_encoding_layers-1): # the number of layers would be n_encoding_layers - 1 for now. # 4
        number_feature = min(base_feature_number * ( 2**(config.n_encoding_layers-i-2) ), config.max_feature_number)
        layer_lower, layer_higher = net, encoding_layers[-(i+2)]

        # upsample
        layer_lower = Conv1D(number_feature, 7, activation = 'relu',
                                padding = 'same',
                                kernel_initializer = 'he_normal',
                                name='decoding_conv_{}'.format(i))(UpSampling1D(size=2, name='decoding_upsample_{}'.format(i))(layer_lower))
        layer_lower = BatchNormalization(name='decoding_bn_{}'.format(i))(layer_lower) if config.batch_normalization else layer_lower

        # crop if needed
        layer_higher = CenterCropLike()([layer_higher, layer_lower]) if config.model_padding == 'valid' else layer_higher

        # merge
        net = concatenate([layer_higher, layer_lower], axis = 2)

    # final output
    for i in range(config.n_final_conv): # 3
        net = Conv1D(base_feature_number, 7, activation = 'relu',
                        padding = 'same',
                        kernel_initializer = 'he_normal',
                        name='final_conv_{}'.format(i))(net)
        net = BatchNormalization(name='final_bn_{}'.format(i))(net) if config.batch_normalization else net


    if config.seg_setting == 'split':
        output_shape = 2
    else: # pqrst
        output_shape = 5 if config.regression else 6

    if config.ending_lstm:
        net = LSTM(units = 8, return_sequences = True)(net)
        net = LSTM(units = 8, return_sequences = True)(net)
        net = LSTM(units = output_shape, return_sequences = True)(net) # spliting cardiac cycle or predict qprst
        net = Softmax()(net)
        output = net
    else:
        output = Conv1D(output_shape, 7,
                        activation = 'linear' if config.regression else 'softmax',
                        padding = 'same', # config.model_padding,
                        kernel_initializer = 'he_normal',
                        name='output_conv')(net)

    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = Adam(lr=1e-4, amsgrad=config.amsgrad),
                                    loss = 'mean_squared_error' if config.regression else 'categorical_crossentropy',
                                    metrics = ['accuracy'], sample_weight_mode="temporal")
    return model, model.layers[-1].output_shape

def unet_1d(config):
    input_size = (5000, 1) if config.target == '24hr' else (10000, 8) # 'bigexam'

    inputs = Input(input_size)
    conv1 = Conv1D(8, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv1D(8, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(16, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv1D(16, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(32, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv1D(32, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(64, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv1D(64, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(128, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv1D(128, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv5)

    for _ in range(config.n_middle_lstm):
        lstm_layer = Bidirectional(LSTM(units=32, return_sequences=True)) if config.bidirectional else LSTM(units=32, return_sequences=True)
        conv5 = lstm_layer(conv5)

    up6 = Conv1D(64, 2, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv5))

    if config.model_padding == 'same' and config.target == '24hr': # since the data length is 5000, which is not divisible by 2^4
        up6 = ZeroPadding1D(padding=(1,0))(up6)

    conv4 = CenterCropLike()([conv4, up6]) if config.model_padding == 'valid' else conv4

    merge6 = concatenate([conv4,up6], axis = 2)
    conv6 = Conv1D(64, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv1D(64, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv6)

    up7 = Conv1D(32, 2, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))

    conv3 = CenterCropLike()([conv3, up7]) if config.model_padding == 'valid' else conv3
    merge7 = concatenate([conv3,up7], axis = 2)
    conv7 = Conv1D(32, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv1D(32, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv7)

    up8 = Conv1D(16, 2, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))

    conv2 = CenterCropLike()([conv2, up8]) if config.model_padding == 'valid' else conv2
    merge8 = concatenate([conv2,up8], axis = 2)
    conv8 = Conv1D(16, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv1D(16, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv8)

    up9 = Conv1D(8, 2, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))

    conv1 = CenterCropLike()([conv1, up9]) if config.model_padding == 'valid' else conv1
    merge9 = concatenate([conv1,up9], axis = 2)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = config.model_padding, kernel_initializer = 'he_normal')(conv9)

    ouptut_shape = 2 if config.seg_setting == 'split' else 6
    if config.ending_lstm:
        conv10 = LSTM(units = 8, return_sequences = True)(conv9)
        conv11 = LSTM(units = 8, return_sequences = True)(conv10)
        conv12 = LSTM(units = ouptut_shape, return_sequences = True)(conv11) # spliting cardiac cycle or predict qprst
        conv13 = Softmax()(conv12)
        output = conv13
    else:
        output = Conv1D(ouptut_shape, 7, activation = 'softmax', padding = config.model_padding, kernel_initializer = 'he_normal')(conv9)

    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = Adam(lr=1e-4, amsgrad=config.amsgrad), loss = 'categorical_crossentropy', metrics = ['accuracy'], sample_weight_mode="temporal")
    return model
