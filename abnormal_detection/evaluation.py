import os, sys
import numpy as np
import sklearn.metrics
import better_exceptions; better_exceptions.hook()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.eval_utils import print_cm
from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()

def evaluation(models, test_set):
    print('Testing set baseline:', 1. - test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0])

    y_pred_score = np.zeros_like(test_set[1][:, 1], dtype=float)
    for model in models:
        y_pred_score += model.predict(test_set[0], batch_size=64)[:, 1]

    y_pred = (y_pred_score > (len(models) * 0.5)).astype(int)

    y_true = test_set[1][:, 1]
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_score)

    print('Total accuracy:', accuracy)
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

    print('ROC AUC:', roc_auc)

    return accuracy, precision, recall, f1_score, roc_auc

def log_evaluation(models, test_set, log_prefix):
    accuracy, precision, recall, f1_score, roc_auc = evaluation(models, test_set)

    wandb.log({'{}_acc'.format(log_prefix): accuracy})
    wandb.log({'{}_precision'.format(log_prefix): precision})
    wandb.log({'{}_recall'.format(log_prefix): recall})
    wandb.log({'{}_f1_score'.format(log_prefix): f1_score})
    wandb.log({'{}_roc_auc'.format(log_prefix): roc_auc})

def print_statistics(train_set, valid_set, test_set):
    print('Statistics:')
    for set_name, dataset in [['Training set', train_set], ['Validation set', valid_set], ['Testing set', test_set]]:
        number_abnormal = dataset[1][:, 1].sum()
        number_normal = dataset[1][:, 0].sum()

        print('{}:'.format(set_name))
        print('\tAbnormal : Normal = {} : {}'.format(number_abnormal, number_normal))
        print('\tAbnormal Ratio: {:.4f}'.format(number_abnormal / (number_normal + number_abnormal)))
        print()

if __name__ == '__main__':
    import wandb
    import configparser

    from ekg.utils.eval_utils import parse_wandb_models
    from ekg.utils.eval_utils import get_evaluation_args, evaluation_log

    from train import AbnormalBigExamLoader, AbnormalAudicor10sLoader
    from train import preprocessing, AbnormalDataGenerator
    from ekg.utils.data_utils import BaseDataGenerator

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config.cfg'))

    # get arguments
    args = get_evaluation_args('Abnormal detection evaluation.')

    # parse models and configs
    models, wandb_configs, model_paths, sweep_name = parse_wandb_models(args.paths, args.n_model, args.metric)
    numbers_models = args.n_model if args.n_model is not None else [len(models)]

    models[0].summary()

    # get data
    wandb_config = wandb_configs[0]
    dataloaders = list()
    if 'big_exam' in wandb_config.datasets:
        dataloaders.append(AbnormalBigExamLoader(wandb_config=wandb_config))
    if 'audicor_10s' in wandb_config.datasets:
        dataloaders.append(AbnormalAudicor10sLoader(wandb_config=wandb_config))

    g = AbnormalDataGenerator(dataloaders=dataloaders,
                            wandb_config=wandb_config,
                            preprocessing_fn=preprocessing)
    train_set, valid_set, test_set = g.get()
    print_statistics(train_set, valid_set, test_set)

    for n_model in numbers_models:
        wandb.init(project='ekg-abnormal_detection', entity=config['General']['wandb_entity'], reinit=True)

        evaluation_log(wandb_configs[:n_model], sweep_name, 
                        args.paths[0] if args.n_model is not None else '',
                        model_paths[:n_model])

        model_set = models[:n_model]

        print('Training set:')
        log_evaluation(model_set, train_set, 'best')

        print('Validation set:')
        log_evaluation(model_set, valid_set, 'best_val')

        print('Testing set:')
        if args.log_test:
            log_evaluation(model_set, test_set, 'best_test')
        else:
            evaluation(model_set, test_set)