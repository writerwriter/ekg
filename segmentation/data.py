import numpy as np
import pandas as pd
import os
import json
import pickle as pkl
import BaselineWanderRemoval as bwr

leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 500
raw_dataset_path = "../data/ecg_data_200.json"
pkl_filename = "../data/dataset_fixed_baseline.pkl"

def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    Y = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])
        
        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)
        
        y.append(p)
        y.append(qrs)
        y.append(t)
        y.append(background)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)

    return {"x" : X, "y" : Y}

def get_mask(table, length):
    mask = [0] * length
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i] == 0 and qrs[i] == 0 and t[i] == 0:
            background[i] = 1
    return background

def fix_baseline_and_save_to_pkl(xy):
    print("start fixing baseline in the whole dataset.")
    X = xy["x"]
    for i in range(X.shape[0]):
        print(str(i))
    
        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy["x"] = X
    outfile = open(pkl_filename, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()
    print("dataset saved, number of patients = " + str(len(xy['x'])))

def load_dataset(raw_dataset=raw_dataset_path, fixed_baseline=True):
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WARNING")
        if os.path.exists(pkl_filename):
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset)
            fix_baseline_and_save_to_pkl(xy)
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
    else:
        print("you selected NOT FIXED BASELINE")
        return load_raw_dataset(raw_dataset)

if __name__ == "__main__":
    xy = load_dataset()
    print(xy['y'].shape)
    print(xy['x'].shape)