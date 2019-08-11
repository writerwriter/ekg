from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, concatenate, UpSampling1D, Softmax, ZeroPadding1D
from keras.models import Model
from keras.optimizers import Adam

def unet_1d(config):
    input_size = (5000, 1) if config.target == '24hr' else (10000, 8) # 'bigexam'

    inputs = Input(input_size)
    conv1 = Conv1D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv1D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv1D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv1D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv1D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(128, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv1D(128, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv5))

    if config.target == '24hr': # since the data length is 5000, which is not divisible by 2^4
        up6 = ZeroPadding1D(padding=(1,0))(up6)

    for _ in range(config.n_middle_lstm):
        up6 = LSTM(units=32, return_sequences=True)(up6)

    merge6 = concatenate([conv4,up6], axis = 2)
    conv6 = Conv1D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv1D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv1D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))
    merge7 = concatenate([conv3,up7], axis = 2)
    conv7 = Conv1D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv1D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv1D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))
    merge8 = concatenate([conv2,up8], axis = 2)
    conv8 = Conv1D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv1D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv1D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))
    merge9 = concatenate([conv1,up9], axis = 2)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    ouptut_shape = 2 if config.seg_setting == 'split' else 6
    if config.ending_lstm:
        conv10 = LSTM(units = 8, return_sequences = True)(conv9)
        conv11 = LSTM(units = 8, return_sequences = True)(conv10)
        conv12 = LSTM(units = ouptut_shape, return_sequences = True)(conv11) # spliting cardiac cycle or predict qprst
        conv13 = Softmax()(conv12)
        output = conv13
    else:
        output = Conv1D(ouptut_shape, 7, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = Adam(lr=1e-4, amsgrad=config.amsgrad), loss = 'categorical_crossentropy', metrics = ['accuracy'], sample_weight_mode="temporal")
    return model
