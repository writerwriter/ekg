import numpy as np
import better_exceptions; better_exceptions.hook()

from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.eval_utils import get_KM_plot, get_survival_scatter

import wandb

def ensemble_predict(models, X, batch_size=64):
    y_preds = list()
    for i, m in enumerate(models):
        y_preds.append(m.predict(X, batch_size=batch_size))

    y_pred = np.stack(y_preds, axis=2).mean(axis=-1) # (?, n_events, num_models) -> (?, n_events)
    return y_pred

def evaluation(models, test_set, event_names, reverse=True):
    '''
        test_set[1]: (num_samples, num_events, 2)
    '''
    X, y = test_set[0], test_set[1]

    try:
        y_pred = ensemble_predict(models, X)

    except Exception:
        y_pred = models.predict(X, batch_size=64)

    cindices = list()

    if reverse: y_pred = y_pred * -1

    # calculate cindex
    for i in range(len(event_names)): # for every event
        event_cs, event_st = y[:, i, 0], y[:, i, 1]
        cindex = concordance_index(event_st, y_pred[:, i], event_cs) # cindex of the event
        print('Concordance index of {} : {:.4f}'.format(event_names[i], cindex))
        cindices.append(cindex)

    return np.array(cindices)

def log_evaluation(models, test_set, log_prefix, event_names, reverse=True):
    cindices = evaluation(models, test_set, event_names, reverse)
    for cindex, event_name in zip(cindices, event_names):
        log_name = '{}_{}_cindex'.format(log_prefix, event_name)
        wandb.log({log_name: cindex})

def evaluation_plot(models, train_set, run_set, prefix='', reverse=True):
    # upload plots
    try:
        train_pred = models.predict(train_set[0])
        run_pred = models.predict(run_set[0])
    except:
        train_pred = ensemble_predict(models, train_set[0])
        run_pred = ensemble_predict(models, run_set[0])

    # KM curve
    for i, event_name in enumerate(wandb.config.events):
        wandb.log({'{}best model {} KM curve'.format(prefix, event_name): 
                        wandb.Image(get_KM_plot(train_pred[:, i], run_pred[:, i], run_set[1][:, i], event_name, reverse=reverse))})
        plt.close()

    # scatter
    for i, event_name in enumerate(wandb.config.events):
        wandb.log({'{}best model {} scatter'.format(prefix, event_name): 
                        wandb.Image(get_survival_scatter(run_pred[:, i], 
                                                        run_set[1][:, i, 0], 
                                                        run_set[1][:, i, 1], 
                                                        event_name, reverse=reverse))})
        plt.close()

def print_statistics(train_set, valid_set, test_set, event_names):
    print('Statistics:')
    for set_name, dataset in [['Training set', train_set], ['Validation set', valid_set], ['Testing set', test_set]]:
        print('{}:'.format(set_name))
        for i, event_name in enumerate(event_names):
            cs = dataset[1][:, i, 0]

            print('{}:'.format(event_name))
            print('\t# of censored:', (cs==0).sum())
            print('\t# of events:', (cs==1).sum())
            print('\tevent ratio: {:.4f}'.format((cs==1).sum() / (cs==0).sum()))
            print()

def to_prediction_model(trainable_model, include_info):
    from tensorflow.keras.models import Model
    # get the outputs of the prediction model
    original_output = trainable_model.get_layer('output')
    return Model([trainable_model.input[:-2]], [original_output.output])

if __name__ == '__main__':
    from train import HazardBigExamLoader, HazardAudicor10sLoader, HazardDataGenerator
    from train import preprocessing

    from ekg.utils.eval_utils import parse_wandb_models
    from ekg.utils.eval_utils import get_evaluation_args, evaluation_log
    from ekg.utils.data_utils import BaseDataGenerator

    wandb.init(project='ekg-hazard_prediction', entity='toosyou')

    # get arguments
    args = get_evaluation_args('Hazard prediction evaluation.')

    # parse models and configs
    models, wandb_configs, model_paths, sweep_name = parse_wandb_models(args.paths, args.n_model, args.metric)

    evaluation_log(wandb_configs, sweep_name, 
                    args.paths[0] if args.n_model >= 1 else '',
                    model_paths)

    models[0].summary()

    # get data
    wandb_config = wandb_configs[0]
    dataloaders = list()
    if 'big_exam' in wandb_config.datasets:
        dataloaders.append(HazardBigExamLoader(wandb_config=wandb_config))
    if 'audicor_10s' in wandb_config.datasets:
        dataloaders.append(HazardAudicor10sLoader(wandb_config=wandb_config))

    g = HazardDataGenerator(do_wavelet=wandb.config.wavelet,
                            dataloaders=dataloaders,
                            wandb_config=wandb_config,
                            preprocessing_fn=preprocessing)

    # convert to prediction_model
    models = [to_prediction_model(model, wandb_config.include_info) for model in models]
    
    reverse = (wandb_config.loss != 'AFT')

    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set, wandb_config.events)

    print('Training set:')
    log_evaluation(models, train_set, 'best', wandb_config.events, reverse)

    print('Validation set:')
    log_evaluation(models, valid_set, 'best_val', wandb_config.events, reverse)

    print('Testing set:')
    evaluation(models, test_set, wandb_config.events, reverse)

    evaluation_plot(models, train_set, train_set, 'training - ', reverse)
    evaluation_plot(models, train_set, valid_set, 'validation - ', reverse)
    evaluation_plot(models, train_set, test_set, 'testing - ', reverse)