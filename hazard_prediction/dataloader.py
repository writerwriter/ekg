import numpy as np
import pandas as pd
import os

from ekg.utils.datasets import BigExamLoader, Audicor10sLoader

def preprocessing(dataloader):
    '''Remove incomplete data
    '''
    remove_mask = np.zeros((dataloader.abnormal_y.shape[0], ), dtype=bool) # all False

    for i, event_name in enumerate(dataloader.config.events):
        remove_mask = np.logical_or(remove_mask, dataloader.abnormal_y[:, i, 0] == -1)

    if dataloader.config.include_info:
        for i, info in enumerate(dataloader.config.infos):
            remove_mask = np.logical_or(remove_mask, dataloader.abnormal_info[:, i] == -1)

    keep_mask = ~remove_mask
    dataloader.abnormal_X = dataloader.abnormal_X[keep_mask]
    dataloader.abnormal_y = dataloader.abnormal_y[keep_mask]
    dataloader.abnormal_subject_id = dataloader.abnormal_subject_id[keep_mask]
    dataloader.abnormal_info = dataloader.abnormal_info[keep_mask]

    # limit censoring for every event
    for i, event_name in enumerate(dataloader.config.events):
        censor_mask = dataloader.abnormal_y[:, i, 1] > dataloader.config.censoring_limit # st > censoring_limit
        dataloader.abnormal_y[censor_mask, i, 0] = 0 # cs: no event occurred
        dataloader.abnormal_y[censor_mask, i, 1] = dataloader.config.censoring_limit # st: censoring_limit

        censor_mask = dataloader.normal_y[:, i, 1] > dataloader.config.censoring_limit # st > censoring_limit
        dataloader.normal_y[censor_mask, i, 0] = 0 # cs: no event occurred
        dataloader.normal_y[censor_mask, i, 1] = dataloader.config.censoring_limit # st: censoring_limit

class HazardBigExamLoader(BigExamLoader):
    def load_abnormal_y(self):
        '''
        Output:
            np.ndarray of shape [n_instances, n_events, 2], where:
                [n_instances, n_events, 0] = cs
                [n_instances, n_events, 1] = st
        '''
        y = np.zeros((self.abnormal_X.shape[0] // len(self.channel_set), len(self.config.events), 2))

        df = pd.read_csv(os.path.join(self.datadir, 'abnormal_event.csv'))
        for i, event_name in enumerate(self.config.events):
            y[:, i, 0] = df['{}_censoring_status'.format(event_name)].values
            y[:, i, 1] = df['{}_survival_time'.format(event_name)].values

        return np.tile(y, [len(self.channel_set), 1, 1])

    def get_split(self, rs=42):
        def has_empty(dataset):
            y = dataset[1]
            for i in range(len(self.config.events)):
                if (y[:, i, 0] == 1).sum() <= 1: # # of signals with events
                    return True
            return False

        datasets = super().get_split(rs)

        # do split again if there's any set with empty events
        if len(self.config.datasets) == 1:
            while True:
                do_again = False
                for dataset in datasets:
                    if has_empty(dataset):
                        do_again = True
                        break
                if do_again:
                    rs += 1
                    datasets = super().get_split(rs)
                else:
                    break # no empty event in any set
        return datasets

class HazardAudicor10sLoader(Audicor10sLoader):
    def load_abnormal_filenames(self):
        '''Return fixed abnormal filenames as DataFrame
        '''
        filenames = np.load(os.path.join(self.datadir, 'abnormal_filenames.npy'))
        filenames = np.vectorize(lambda fn: fn.split('/')[-1].lower())(filenames)
        filenames = np.vectorize(lambda fn: fn.replace('_v_', '_v1_'))(filenames)               # fix FEMH020_V_Snapshot.txt
        filenames = np.vectorize(lambda fn: fn.replace('_screen_', '_screening_'))(filenames)   # fix FEMH026_SCREEN_Snapshot.txt
        filenames = np.vectorize(lambda fn: fn.replace('_screeing_', '_screening_'))(filenames) # fix TVGH066_SCREEING_Snapshot.txt

        return pd.DataFrame(filenames, columns=['filename'])

    def load_abnormal_info(self):
        df = pd.read_csv(os.path.join(self.datadir, 'abnormal_event.csv'))
        df.filename = df.filename.str.lower()   # make sure both SCREENING / screening work

        # load filenames of abnormal data
        filename_df = self.load_abnormal_filenames()

        # use the filename to get info
        merged_df = pd.merge(filename_df, 
                                df[['filename'] + self.config.infos],
                                left_on='filename', right_on='filename',
                                how='left')
        merged_df = merged_df.replace(np.nan, -1) # replace nan with -1
        infos = merged_df[self.config.infos].values
        return infos

    def load_abnormal_y(self):
        '''
        Output:
            np.ndarray of shape [n_instances, n_events, 2], where:
                [n_instances, n_events, 0] = cs
                [n_instances, n_events, 1] = st
        '''
        y = np.zeros((self.abnormal_X.shape[0] // len(self.channel_set), len(self.config.events), 2))

        df = pd.read_csv(os.path.join(self.datadir, 'abnormal_event.csv'))
        df.filename = df.filename.str.lower()   # make sure both SCREENING / screening work

        # load filenames of abnormal data
        filename_df = self.load_abnormal_filenames()

        # use the filename to get cs and st
        for i, event_name in enumerate(self.config.events):
            merged_df = pd.merge(filename_df, 
                                    df[['filename', '{}_censoring_status'.format(event_name), '{}_survival_time'.format(event_name)]],
                                    left_on='filename', right_on='filename',
                                    how='left')
            merged_df = merged_df.replace(np.nan, -1) # replace nan with -1
            y[:, i, 0] = merged_df['{}_censoring_status'.format(event_name)]
            y[:, i, 1] = merged_df['{}_survival_time'.format(event_name)]

        return np.tile(y, [len(self.channel_set), 1, 1])

def load_normal_y(self):
    '''
    Output:
        np.ndarray of shape [n_instances, n_events, 2], where:
            [n_instances, n_events, 0] = cs, 0 survived
            [n_instances, n_events, 1] = st, maximum value of that events
    '''
    y = np.zeros((self.normal_X.shape[0], len(self.config.events), 2))
    y[:, :, 0] = 0 # survived
    y[:, :, 1] = self.abnormal_y[:, :, 1].max()
    
    return y

HazardBigExamLoader.load_normal_y = load_normal_y
HazardAudicor10sLoader.load_normal_y = load_normal_y