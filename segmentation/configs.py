original_unet = {
    'target': '24hr',
    'seg_setting': 'pqrst',

    'amsgrad': True,  #TODO: Change optimizer to radam
    'n_encoding_layers': 5,
    'n_initial_layers': 0,
    'n_conv_per_encoding_layer': 2,
    'kernel_size_encoding': 7,
    'index_middle_lstm': 5,
    'n_middle_lstm': 0,
    'n_final_conv': 0,
    'base_feature_number': 8,
    'max_feature_number': 64,
    'ending_lstm': True,
    'model_padding': 'same',
    'bidirectional': False,
    'batch_normalization': False,

    # data
    'peak_weight': 0,
    'window_moving_average': 0,
    'window_weight_forgiveness': 0,

    'regression': True,
    'label_blur_kernel': 11,
    'label_normalization': True,
    'label_normalization_value': 512, # TODO: calculate it by peak and background ratio
    'use_all_leads': False,
}
