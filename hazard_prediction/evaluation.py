import numpy as np
import better_exceptions; better_exceptions.hook()

from lifelines.utils import concordance_index

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

    # calculate cindex
    for i in range(len(event_names)): # for every event
        event_cs, event_st = y[:, i, 0], y[:, i, 1]
        cindex = concordance_index(event_st, -y_pred[:, i], event_cs) # cindex of the event
        print('Concordance index of {} : {:.4f}'.format(event_names[i], cindex))

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

def dict_to_wandb_config(config):
    class Object(object):
        pass

    wandb_config = Object()
    for key, value in config.items():
        setattr(wandb_config, key, value)
    return wandb_config

if __name__ == '__main__':
    import argparse
    import wandb

    from keras.models import load_model

    from train import HazardBigExamLoader, HazardAudicor10sLoader
    from train import preprocessing

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    from ekg.layers import LeftCropLike
    from ekg.layers.sincnet import SincConv1D
    from ekg.layers import CenterCropLike

    from ekg.utils import train_utils; train_utils.allow_gpu_growth()
    from ekg.utils.data_utils import BaseDataGenerator

    parser = argparse.ArgumentParser(description='Hazard prediction evaluation.')
    parser.add_argument('run_paths', metavar='run_paths', type=str, nargs='+',
                        help='Run paths of wandb to be evaluated.')

    args = parser.parse_args()

    # parse models and configs
    api = wandb.Api()
    models, wandb_configs = parse_runs(api, args.run_paths)
    wandb_config = dict_to_wandb_config(wandb_configs[0])

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
    evaluation(models, test_set, wandb_config.events)
