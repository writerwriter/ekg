import copy
import pprint
from itertools import product

base_setting = {
    'method': 'bayes',

    'metric':{
        'name': 'best_val_loss',
        'goal': 'minimize'
    },

    'early_terminate':{
        'type': 'hyperband'
    },

    'parameters':{
        'sincconv_filter_length':{
            'min': 16,
            'max': 128
        },
        'sincconv_nfilters':{
            'values': [8, 16, 32]
        },
        'batch_size': {
            'value': 64
        },
        'branch_nlayers':{
            'values': [1, 2, 3, 4, 5]
        },
        'ekg_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'hs_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'ekg_nfilters':{
            'values': [1, 2, 4, 8, 16, 32]
        },
        'hs_nfilters':{
            'values': [1, 2, 4, 8, 16, 32]
        },
        'final_nlayers':{
            'values': [3, 4, 5, 6]
        },
        'final_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'final_nonlocal_nlayers':{
            'value': 0
        },
        'final_nfilters':{
            'values': [8, 16, 32]
        },
        'kernel_initializer':{
            'values': ['glorot_uniform'] # , 'he_normal']
        },
        'skip_connection':{
            'values': [True, False]
        },
        'crop_center':{
            'values': [True]
        },
        'se_block':{
            'values': [True, False]
        },
        'radam':{
            'values': [True, False]
        }
    }
}

def set_parameters(d, key, value, search=False):
    d['parameters'][key] = {
        'values' if search else 'value': copy.deepcopy(value)
    }

def set_parameters_range(d, key, min, max):
    d['parameters'][key] = {
        'min': min,
        'max': max
    }

def generate_sweep(task, dataset, hs_ekg_setting, with_normal=True, survival_model=None, AFT_distribution=None):
    sweep = copy.deepcopy(base_setting)
    sweep['name'] = '{}/{}/{}/{}'.format(task, dataset, hs_ekg_setting, 'with_normal' if with_normal else 'without_normal')
    sweep['program'] = './{}/train.py'.format(task)

    set_parameters(sweep, 'datasets', [dataset] if 'hybrid' not in dataset else ['audicor_10s', 'big_exam'])
    
    # audicor_10s
    set_parameters(sweep, 'audicor_10s_ekg_channels', list() if hs_ekg_setting == 'only_hs' else [0])
    set_parameters(sweep, 'audicor_10s_hs_channels', list() if hs_ekg_setting == 'only_ekg' else [1])
    set_parameters(sweep, 'audicor_10s_only_train', False)

    # big_exam
    if 'hybrid' in dataset:
        set_parameters(sweep, 'big_exam_ekg_channels', list() if hs_ekg_setting == 'only_hs' else [1])
    else:
        set_parameters(sweep, 'big_exam_ekg_channels', list() if hs_ekg_setting == 'only_hs' else list(range(8)))

    set_parameters(sweep, 'big_exam_hs_channels', list() if hs_ekg_setting == 'only_ekg' else [8, 9])
    set_parameters(sweep, 'big_exam_only_train', (dataset == 'hybrid/audicor_as_test'))

    set_parameters(sweep, 'with_normal_subjects', with_normal)
    set_parameters(sweep, 'normal_subjects_only_train', False)

    # hazard_prediction
    if task == 'hazard_prediction':

        # model
        set_parameters(sweep, 'prediction_head', [True, False], search=True)
        set_parameters(sweep, 'prediction_nlayers', [2, 3, 4, 5], search=True)
        set_parameters(sweep, 'prediction_kernel_length', [5, 7, 13, 21, 35], search=True)
        set_parameters(sweep, 'prediction_nfilters', [8, 16, 32], search=True)

        # data
        events = ['ADHF', 'Mortality'] + (['MI', 'CVD'] if dataset == 'big_exam' else []) # Stoke
        set_parameters(sweep, 'events', events)
        set_parameters(sweep, 'event_weights', [1 for _ in events])
        set_parameters(sweep, 'censoring_limit', 400 if dataset == 'hybrid/audicor_as_test' else 99999)

        sweep['name'] = '{}/{}'.format(sweep['name'], survival_model)
        set_parameters(sweep, 'loss', survival_model) # AFT or Cox
        if survival_model == 'AFT':
            sweep['parameters']['radam'] = {
                'value': False
            }
            sweep['name'] = '{}/{}'.format(sweep['name'], AFT_distribution)
            set_parameters(sweep, 'AFT_distribution', AFT_distribution)
            set_parameters_range(sweep, 'AFT_initial_sigma', 0.3, 1.0)

        set_parameters(sweep, 'include_info', not with_normal) # without normal -> include info
        if not with_normal: # include info
            set_parameters(sweep, 'infos', ['sex', 'age', 'BMI']) # , 'height', 'weight'
            set_parameters(sweep, 'info_noise_stds', [0, 1, 0.5]) # [0, 1, 1, 1, 0.25]
            set_parameters(sweep, 'info_apply_noise', True)

            set_parameters(sweep, 'info_nlayers', [1, 2, 3, 4, 5], search=True)
            set_parameters(sweep, 'info_units', [8, 16, 32, 64], search=True)
    
    return sweep

def get_all_sweeps():
    sweeps = list()
    tasks = ['abnormal_detection', 'hazard_prediction']
    datasets = ['audicor_10s', 'big_exam', 'hybrid/audicor_as_test', 'hybrid/both_as_test']
    hs_ekg_settings = ['only_hs', 'only_ekg', 'whole']

    # hazard prediction only
    normal_settings = ['with_normal', 'without_normal']
    survival_models = ['Cox', 'AFT']

    # AFT only
    AFT_distributions = ['weibull', 'log-logistic']

    # abnormal detection
    for task, dataset, hs_ekg_setting in product(tasks[:1], datasets, hs_ekg_settings):
        sweeps.append(generate_sweep(task, dataset, hs_ekg_setting))

    # hazard prediction
    for task, dataset, hs_ekg_setting, normal_setting, survival_model in product(tasks[1:], datasets, hs_ekg_settings, normal_settings, survival_models):
        if survival_model == 'AFT':
            for distribution in AFT_distributions:
                sweeps.append(generate_sweep(task, dataset, hs_ekg_setting, (normal_setting == 'with_normal'), survival_model, distribution))
        else:
            sweeps.append(generate_sweep(task, dataset, hs_ekg_setting, (normal_setting == 'with_normal'), survival_model))

    return sweeps

if __name__ == '__main__':
    pprint.pprint(get_all_sweeps())