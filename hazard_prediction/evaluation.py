import numpy as np
import better_exceptions; better_exceptions.hook()

from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.eval_utils import get_KM_plot, get_survival_scatter
from ekg.utils.train_utils import set_wandb_config

def ensemble_predict(models, X, batch_size=64):
    y_preds = list()
    for i, m in enumerate(models):
        y_preds.append(m.predict(X, batch_size=batch_size))

    y_pred = np.stack(y_preds, axis=2).mean(axis=-1) # (?, n_events, num_models) -> (?, n_events)
    return y_pred

def evaluation(models, test_set, event_names):
    '''
        test_set[1]: (num_samples, num_events, 2)
    '''
    X, y = test_set[0], test_set[1]

    try:
        y_pred = ensemble_predict(models, X)

    except Exception:
        y_pred = models.predict(X, batch_size=64)

    cindices = list()

    # calculate cindex
    for i in range(len(event_names)): # for every event
        event_cs, event_st = y[:, i, 0], y[:, i, 1]
        cindex = concordance_index(event_st, -y_pred[:, i], event_cs) # cindex of the event
        print('Concordance index of {} : {:.4f}'.format(event_names[i], cindex))
        cindices.append(cindex)

    return np.array(cindices)

def log_evaluation(models, test_set, log_prefix, event_names):
    cindices = evaluation(models, test_set, event_names)
    for cindex, event_name in zip(cindices, event_names):
        log_name = '{}_{}_cindex'.format(log_prefix, event_name)
        wandb.log({log_name: cindex})

def parse_runs(api, run_paths):
    models, configs = list(), list()

    modeldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(modeldir, exist_ok=True)
    for run_path in run_paths:
        run = api.run(run_path)
        run.file('model-best.h5').download(replace=True, root=modeldir)

        models.append(load_model(modeldir + '/model-best.h5', 
                            custom_objects={'SincConv1D': SincConv1D,
                                            'LeftCropLike': LeftCropLike,
                                            'CenterCropLike': CenterCropLike}, compile=False))
        configs.append(run.config)
    return models, configs

def parse_sweep(api, sweep_path, number_models, metric):
    models, configs, model_paths = list(), list(), list()

    modeldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(modeldir, exist_ok=True)

    sweep = api.sweep(sweep_path[0])
    # sort runs by metric
    runs = sorted(sweep.runs, key=lambda run: run.summary.get(metric, np.Inf if 'loss' in metric else 0), 
                        reverse=False if 'loss' in metric else True)
    for best_run in runs[:number_models]:
        best_run.file('model-best.h5').download(replace=True, root=modeldir)

        # load model
        models.append(load_model(modeldir + '/model-best.h5', 
                            custom_objects={'SincConv1D': SincConv1D,
                                            'LeftCropLike': LeftCropLike,
                                            'CenterCropLike': CenterCropLike}, compile=False))
        configs.append(best_run.config)
        model_paths.append(best_run.path)
    return models, configs, model_paths

def dict_to_wandb_config(config):
    class Object(object):
        pass

    wandb_config = Object()
    for key, value in config.items():
        setattr(wandb_config, key, value)
    return wandb_config

def evaluation_plot(models, train_set, run_set, prefix=''):
    import wandb

    # upload plots
    try:
        train_pred = model.predict(train_set[0])
        run_pred = model.predict(run_set[0])
    except:
        train_pred = ensemble_predict(models, train_set[0])
        run_pred = ensemble_predict(models, run_set[0])

    # KM curve
    for i, event_name in enumerate(wandb.config.events):
        wandb.log({'{}best model {} KM curve'.format(prefix, event_name): 
                        wandb.Image(get_KM_plot(train_pred[:, i], run_pred[:, i], run_set[1][:, i], event_name))})
        plt.close()

    # scatter
    for i, event_name in enumerate(wandb.config.events):
        wandb.log({'{}best model {} scatter'.format(prefix, event_name): 
                        wandb.Image(get_survival_scatter(run_pred[:, i], 
                                                        run_set[1][:, i, 0], 
                                                        run_set[1][:, i, 1], 
                                                        event_name))})
        plt.close()

def print_statistics(train_set, valid_set, test_set):
    print('Statistics:')
    for set_name, dataset in [['Training set', train_set], ['Validation set', valid_set], ['Testing set', test_set]]:
        print('{}:'.format(set_name))
        for i, event_name in enumerate(wandb.config.events):
            cs = dataset[1][:, i, 0]

            print('{}:'.format(event_name))
            print('\t# of censored:', (cs==0).sum())
            print('\t# of events:', (cs==1).sum())
            print('\tevent ratio: {:.4f}'.format((cs==1).sum() / (cs==0).sum()))
            print()

def log_config(configs):
    def all_same(check_key, check_value, configs):
        for config in configs:
            if config[check_key] != check_value:
                return False
        return True
                    

    if len(configs) == 1:
        set_wandb_config(configs)
        return

    # get the key with the same value across configs
    for key, value in configs[0].items():
        if all_same(key, value, configs):
            set_wandb_config({key: value})

if __name__ == '__main__':
    import argparse
    import wandb

    from tensorflow.keras.models import load_model

    from train import HazardBigExamLoader, HazardAudicor10sLoader
    from train import preprocessing

    from ekg.layers import LeftCropLike
    from ekg.layers.sincnet import SincConv1D
    from ekg.layers import CenterCropLike

    from ekg.utils import train_utils; train_utils.allow_gpu_growth()
    from ekg.utils.data_utils import BaseDataGenerator

    parser = argparse.ArgumentParser(description='Hazard prediction evaluation.')
    parser.add_argument('-n', '--n_model', type=int, default=-1,
                            help='Number of best models to evaluate.')
    parser.add_argument('-m', '--metric', type=str, default='best_val_loss',
                            help='Which metric to use for selecting best models from the sweep.')
    parser.add_argument('paths', metavar='paths', type=str, nargs='+',
                        help='Run paths or a sweep path of wandb to be evaluated. If n_model >= 1, it will be treated as sweep path.')

    args = parser.parse_args()

    # parse models and configs
    api = wandb.Api()
    if args.n_model == -1:
        models, wandb_configs = parse_runs(api, args.paths)
        wandb_config = dict_to_wandb_config(wandb_configs[0])

        # log models used
        wandb.config.n_models = len(args.run_paths)
        wandb.config.models = args.run_paths

    else: # sweep
        models, wandb_configs, model_paths = parse_sweep(api, args.paths, args.n_model, args.metric)
        wandb_config = dict_to_wandb_config(wandb_configs[0])

        # log models used
        wandb.config.n_models = args.n_model
        wandb.config.models = model_paths

    models[0].summary()

    dataloaders = list()
    if 'big_exam' in wandb_config.datasets:
        dataloaders.append(HazardBigExamLoader(wandb_config=wandb_config))
    if 'audicor_10s' in wandb_config.datasets:
        dataloaders.append(HazardAudicor10sLoader(wandb_config=wandb_config))

    g = BaseDataGenerator(dataloaders=dataloaders,
                            wandb_config=wandb_config,
                            preprocessing_fn=preprocessing)

    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set)

    wandb.config.evaluation = True

    print('Training set:')
    log_evaluation(models, train_set, 'best', wandb_config.events)

    print('Validation set:')
    log_evaluation(models, valid_set, 'best_val', wandb_config.events)

    print('Testing set:')
    evaluation(models, test_set, wandb_config.events)

    evaluation_plot(models, train_set, train_set, 'training - ')
    evaluation_plot(models, train_set, valid_set, 'validation - ')
    evaluation_plot(models, train_set, test_set, 'testing - ')

    # remove the tmp file
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model-best.h5'))
    os.rmdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))