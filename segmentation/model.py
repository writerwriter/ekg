from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, concatenate, UpSampling1D, Softmax, ZeroPadding1D, Bidirectional, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ekg.layers import CenterCropLike, PaddingLike

def unet_lstm(config):
    input_size = (5000, 1) if config.target == '24hr' else (10000, 8) # 'bigexam'

    input = Input(input_size)
    base_feature_number = config.base_feature_number

    net = input

    # initial layers
    for i in range(config.n_initial_layers):
        net = Conv1D(8, 7,
                        activation = 'relu',
                        padding = config.model_padding,
                        kernel_initializer = 'he_normal',
                        name='initial_conv_{}'.format(i))(net)
        net = BatchNormalization(name='initial_bn_{}'.format(i))(net) if config.batch_normalization else net


    # encoding
    encoding_layers = list()
    for i in range(config.n_encoding_layers): # 5
        number_feature = min(base_feature_number*(2**i), config.max_feature_number)
        for j in range(config.n_conv_per_encoding_layer):
            index_encoding_layer = i*config.n_conv_per_encoding_layer + j
            net = Conv1D(number_feature, config.kernel_size_encoding,
                            activation = 'relu',
                            padding = config.model_padding,
                            kernel_initializer = 'he_normal',
                            name='encoding_conv_{}'.format(index_encoding_layer))(net)
            net = BatchNormalization(name='encoding_bn_{}'.format(index_encoding_layer))(net) if config.batch_normalization else net
        encoding_layers.append(net)

        # middle lstms
        if i == config.index_middle_lstm:
            for _ in range(config.n_middle_lstm):
                lstm_layer = Bidirectional(LSTM(units=32, return_sequences=True)) if config.bidirectional else LSTM(units=32, return_sequences=True)
                net = lstm_layer(net)



        if i != config.n_encoding_layers - 1:
            net = MaxPooling1D(pool_size=2, name='encoding_maxpool_{}'.format(i))(net)



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
        layer_lower = PaddingLike()([layer_lower, layer_higher]) if config.model_padding == 'same' else layer_lower

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
