import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D, ReLU, Bidirectional, Maximum
from keras.layers import Conv1D, MaxPooling1D, Dense, Add, Concatenate, Flatten, Dropout, LSTM, Reshape, SeparableConv1D
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

from non_local import non_local_block

from sincnet import SincConv1D

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def get_normal_model():
    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = Conv1D(8, 5, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    ekg = Lambda(lambda x: x[:, 5:-5])(ekg) # (?, 1657, 64)

    hs_outputs = list()
    for hs_i in range(2):
        hs = Lambda(lambda x: x[:, :, hs_i:hs_i+1])(heart_sound_input)

        hs = SincConv1D(8, 63, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs_outputs.append(hs)

    hs = Add()(hs_outputs) # (?, 1625, 128)

    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 208

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = GlobalAveragePooling1D()(output)
    # output = Flatten()(output)
    # output = Dense(8, activation='relu')(output)
    # output = BatchNormalization()(output)
    # output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(2, activation='softmax')(output)

    model = Model(total_input, output)
    model.compile('SGD', 'binary_crossentropy', metrics=['acc'])
    return model

def get_normal_modelV2(expand_loop=False):
    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    # heart sound branch
    hs_outputs = list()

    sincconv_filter_length = 63
    def heart_sound_branch(hs):
        hs = SincConv1D(8, sincconv_filter_length, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        return hs

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 0], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 1], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Add()(hs_outputs) # (?, 1625, 128)
    # hs = Maximum()(hs_outputs)
    # hs = hs_outputs[0]
    # hs = Concatenate(axis=-1)(hs_outputs)

    number_points_ekg = 10000 // 3 //2
    number_points_hs = (10000 - sincconv_filter_length + 1) // 3 // 2

    left_crop = (number_points_ekg - number_points_hs) // 2
    right_crop = (number_points_ekg - number_points_hs) - left_crop
    ekg = Lambda(lambda x: x[:, left_crop:-right_crop])(ekg) # (?, 1657, 64)
    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 208

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = non_local_block(output, compression=2, mode='embedded')
    output = MaxPooling1D(2, padding='same')(output) # 26

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = non_local_block(output, compression=1, mode='embedded')
    output = GlobalAveragePooling1D()(output)

    output = Dense(2, activation='softmax')(output)

    model = Model(total_input, output)
    # model.compile('sgd', 'binary_crossentropy', metrics=['acc'])
    model.compile(Adam(amsgrad=True), 'binary_crossentropy', metrics=['acc'])
    return model

def get_survival_rate_model():
    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = Conv1D(8, 5, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    ekg = Lambda(lambda x: x[:, 5:-5])(ekg) # (?, 1657, 64)

    hs_outputs = list()
    for hs_i in range(2):
        hs = Lambda(lambda x: x[:, :, hs_i:hs_i+1])(heart_sound_input)

        hs = SincConv1D(8, 63, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs_outputs.append(hs)

    hs = Add()(hs_outputs) # (?, 1625, 128)

    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    '''
    output = MaxPooling1D(2, padding='same')(output) # 208

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    '''
    output = GlobalAveragePooling1D()(output)

    # output = Flatten()(output)
    # output = Dense(8, activation='relu')(output)
    # output = BatchNormalization()(output)
    # output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(4, activation='sigmoid')(output)

    model = Model(total_input, output)
    model.compile('SGD', 'binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def get_survival_rate_modelV2():
    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = Conv1D(8, 5, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    ekg = Lambda(lambda x: x[:, 5:-5])(ekg) # (?, 1657, 64)

    hs_outputs = list()
    for hs_i in range(2):
        hs = Lambda(lambda x: x[:, :, hs_i:hs_i+1])(heart_sound_input)

        hs = SincConv1D(8, 63, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs_outputs.append(hs)

    hs = Add()(hs_outputs) # (?, 1625, 128)

    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 208

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 26

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = non_local_block(output, compression=1, mode='embedded')
    output = GlobalAveragePooling1D()(output)

    output = Dense(4, activation='sigmoid')(output)

    model = Model(total_input, output)
    model.compile('SGD', 'binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def get_survival_rate_wb_model():
    blood_input = Input((65, ))
    blood = blood_input

    ekg_hs_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(ekg_hs_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(ekg_hs_input) # (10000, 2)

    ekg = SeparableConv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = SeparableConv1D(4, 5, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    ekg = Lambda(lambda x: x[:, 5:-5])(ekg) # (?, 1657, 64)

    hs_outputs = list()
    for hs_i in range(2):
        hs = Lambda(lambda x: x[:, :, hs_i:hs_i+1])(heart_sound_input)

        hs = SincConv1D(8, 63, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = SeparableConv1D(4, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs_outputs.append(hs)

    hs = Add()(hs_outputs) # (?, 1625, 128)

    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = SeparableConv1D(4, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = SeparableConv1D(4, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = SeparableConv1D(4, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 208

    '''
    output = SeparableConv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = SeparableConv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = SeparableConv1D(8, 7, activation='relu', padding='same')(output)
    '''
    output = GlobalAveragePooling1D()(output)

    # output = Flatten()(output)
    output = Concatenate(axis=-1)([output, blood]) # (52) + (8)
    output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(4, activation='sigmoid')(output)

    model = Model([ekg_hs_input, blood_input], output)
    model.compile('SGD', 'binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def get_survival_hazard_model():
    def _negative_log_likelihood(y, risk):
        E = y[:, 0:1]

        hazard_ratio = K.exp(risk)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood

    def cindex_loss(y, risk):
        cs, st = tf.cast(y[:, 0:1], tf.float32), tf.cast(y[:, 1:2], tf.float32)

        risk_comparison_matrix = tf.subtract(tf.expand_dims(risk, -1), risk)

        risk_larger = K.softsign(risk_comparison_matrix) + 1
        risk_equal = tf.cast(tf.abs(risk_comparison_matrix) < 1e-3, tf.float32) * 0.5
        time_comparison = tf.cast(tf.subtract(tf.expand_dims(st, -1), st) < 0.0, tf.float32)
        ratio = tf.reduce_sum( (tf.reduce_sum(risk_larger * time_comparison, 1) + tf.reduce_sum(risk_equal * time_comparison, 1))*cs ) / tf.reduce_sum(tf.reduce_sum(time_comparison, 1) * cs)
        return -ratio

    def cindex(y, risk):
        cs, st = tf.cast(y[:, 0:1], tf.float32), tf.cast(y[:, 1:2], tf.float32)

        risk_comparison_matrix = tf.subtract(tf.expand_dims(risk, -1), risk)

        risk_larger = tf.cast(risk_comparison_matrix > 0.0, tf.float32)
        risk_equal = tf.cast(tf.abs(risk_comparison_matrix) < 1e-3, tf.float32) * 0.5
        time_comparison = tf.cast(tf.subtract(tf.expand_dims(st, -1), st) < 0.0, tf.float32)
        ratio = tf.reduce_sum( (tf.reduce_sum(risk_larger * time_comparison, 1) + tf.reduce_sum(risk_equal * time_comparison, 1))*cs ) / tf.reduce_sum(tf.reduce_sum(time_comparison, 1) * cs)
        return ratio

    total_input = Input((10000, 10))
    ekg_input = Lambda(lambda x: x[:, :, :8])(total_input) # (10000, 8)
    heart_sound_input = Lambda(lambda x: x[:, :, 8:])(total_input) # (10000, 2)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg_input)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(3, padding='same')(ekg)

    ekg = Conv1D(8, 7, activation='relu', padding='same')(ekg)
    ekg = BatchNormalization()(ekg)
    ekg = MaxPooling1D(2, padding='same')(ekg) # (?, 1666, 64)

    # heart sound branch
    hs_outputs = list()

    sincconv_filter_length = 63
    def heart_sound_branch(hs):
        hs = SincConv1D(8, sincconv_filter_length, 1000)(hs)
        hs = BatchNormalization()(hs)
        hs = ReLU()(hs)
        hs = MaxPooling1D(3, padding='same')(hs) # (?, 3250, 128)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        hs = MaxPooling1D(2, padding='same')(hs)

        hs = Conv1D(8, 7, activation='relu', padding='same')(hs)
        hs = BatchNormalization()(hs)
        return hs

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 0], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Lambda(lambda x: K.expand_dims(x[:, :, 1], -1))(heart_sound_input)
    hs_outputs.append(heart_sound_branch(hs))

    hs = Add()(hs_outputs) # (?, 1625, 128)
    # hs = Maximum()(hs_outputs)
    # hs = hs_outputs[0]
    # hs = Concatenate(axis=-1)(hs_outputs)

    number_points_ekg = 10000 // 3 //2
    number_points_hs = (10000 - sincconv_filter_length + 1) // 3 // 2

    left_crop = (number_points_ekg - number_points_hs) // 2
    right_crop = (number_points_ekg - number_points_hs) - left_crop
    ekg = Lambda(lambda x: x[:, left_crop:-right_crop])(ekg) # (?, 1657, 64)
    output = Concatenate(axis=-1)([hs, ekg])
    # output = hs
    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 829

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 415

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 208

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 104

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    output = MaxPooling1D(2, padding='same')(output) # 52

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    # output = non_local_block(output, compression=2, mode='embedded')
    output = MaxPooling1D(2, padding='same')(output) # 26

    output = Conv1D(8, 7, activation='relu', padding='same')(output)
    output = BatchNormalization()(output)
    # output = non_local_block(output, compression=1, mode='embedded')
    output = GlobalAveragePooling1D()(output)

    output = BatchNormalization()(output)
    output = Dense(1, activation='linear')(output)

    model = Model(total_input, output)
    model.compile(Adam(amsgrad=True), loss=cindex_loss, metrics=[cindex]) # _negative_log_likelihood
    return model

if __name__ == '__main__':
    get_survival_hazard_model().summary()
