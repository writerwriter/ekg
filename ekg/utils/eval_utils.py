import yaml
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

import matplotlib.pyplot as plt

class YamlParser():
    def __init__(self):
        self.config = None

    def read(self, filename):
        with open(filename, 'r') as f:
            self.config = yaml.safe_load(f)

        for key, value in self.config.items():
            setattr(self, key, value)

        return self

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    '''pretty print for confusion matrixes'''
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = ' ' * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * ' ' + 't/p' + (columnwidth-3)//2 * ' '

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = ' ' * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print('    ' + fst_empty_cell, end=' ')
    # End CHANGES

    for label in labels:
        print('%{0}s'.format(columnwidth) % label, end=' ')

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print('    %{0}s'.format(columnwidth) % label1, end=' ')
        for j in range(len(labels)):
            cell = '%{0}.1f'.format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()

def get_KM_plot(train_pred, test_pred, test_true, event_names):
    '''
    Args:
        train_pred:     np.array of shape (n_samples, n_events)
        test_pred:      np.array of shape (n_samples, n_events)
        test_true:      np.array of shape (n_samples, n_events, 2)
                        [:, :, 0] - censoring states
                        [:, :, 1] - survival times
    '''
    # find median of training set
    medians = np.median(train_pred, axis=0)

    for i in range(len(event_names)):
        # split testing data into 2 groups by median, high risk / low risk
        median = medians[i]
        
        high_risk_indices = np.where(test_pred[:, i] >= median)[0]
        low_risk_indices = np.where(test_pred[:, i] < median)[0]
        
        high_risk_cs = test_true[high_risk_indices, i, 0]
        high_risk_st = test_true[high_risk_indices, i, 1]
        
        low_risk_cs = test_true[low_risk_indices, i, 0]
        low_risk_st = test_true[low_risk_indices, i, 1]
        
        # calculate logrank p value
        p_value = logrank_test(high_risk_st, low_risk_st, high_risk_cs, low_risk_cs).p_value
        
        # plot KM curve
        plt.figure(figsize=(20, 10))

        kmf = KaplanMeierFitter()
        kmf.fit(high_risk_st, high_risk_cs, label='high risk')
        a1 = kmf.plot(figsize=(20, 10), title='{} KM curve, logrank p-value: {}'.format(event_names[i], p_value))

        kmf.fit(low_risk_st, low_risk_cs, label='low risk')
        kmf.plot(ax=a1)
        plt.tight_layout()

    figures = list(map(plt.figure, plt.get_fignums()))
    return figures

def get_survival_scatter(y_pred, cs_true, st_true, event_name):
    '''
    Args:
        y_pred: np.array of shape (n_samples)
        cs_true: np.array of shape (n_samples)
        st_true: np.array of shape (n_samples)
    '''
    plt.figure(figsize=(20, 10))

    # plot normal
    normal_mask = (cs_true == 0)
    plt.scatter(y_pred[normal_mask], st_true[normal_mask], color='black', alpha=0.2, label='censored')

    # plot normal
    abnormal_mask = (cs_true == 1)
    plt.scatter(y_pred[abnormal_mask], st_true[abnormal_mask], marker=6, s=100, c='#ff1a1a', label='event occured')

    plt.xlabel('predicted risk')
    plt.ylabel('survival time (days)')
    
    plt.legend()
    plt.title('{} - cindex: {:.3f}'.format(event_name, concordance_index(st_true, -y_pred, cs_true)))

    plt.tight_layout()
    return plt