import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import warnings
import configparser

from tqdm import tqdm

import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ekg.audicor_reader import denoise

config = configparser.ConfigParser()
config.read('./config.cfg')

def outlier_removal(signal, outlier_value=-32768):
    '''Remove the suddent outliers (which have the values of -32768) from the signal inplace.
    args:
        signal: 2d-array of shape (?, signal_length)

    output:
        clean signal
    '''
    for i in range(signal.shape[0]):
        to_be_removed, x = (signal[i] == outlier_value), lambda z: z.nonzero()[0]
        signal[i, to_be_removed]= np.interp(x(to_be_removed), x(~to_be_removed), signal[i, ~to_be_removed])
    
    return signal

def read_data(normal_dir, abnormal_dir, do_bandpass_filter, filter_lowcut, filter_highcut, fs):
    '''Read both normal and abnormal data and do data preprocessing

    output:
        normal
        abnormal: np.ndarray of shape [n_instances, n_channels, n_samples]
                    Note that n_channels is fixed to 2, and signal_length is fixed to 10 seconds for now.
        normal_filenames
        abnormal_filenames: np.array of shape [n_instances]
    '''
    def read_normal(filenames):
        X = np.zeros((len(filenames), 2, fs*10))
        for i, fn in enumerate(tqdm(filenames, desc='reading normal data')):
            with open(fn, 'r') as f:
                for index_line, line in enumerate(f.readlines()):
                    X[i, 0, index_line] = int(line.split()[0]) # ekg
                    X[i, 1, index_line] = int(line.split()[1]) # hs

            # data preprocessing
            X[i] = outlier_removal(X[i])
            X[i, 0:1, :] = denoise.ekg_denoise(X[i, 0:1, :], number_channels=1)
            if do_bandpass_filter:
                X[i, 1:, :] = denoise.heart_sound_denoise(X[i, 1:, :], filter_lowcut, filter_highcut, fs)
        return X
    
    def read_abnormal(filenames):
        X = np.zeros((len(filenames), 2, fs*10))
        for i, fn in enumerate(tqdm(filenames, desc='reading abnormal data')):
            values = pd.read_json(fn).values.reshape(-1)

            # data preprocessing
            ekg = denoise.ekg_denoise(outlier_removal(values[np.newaxis, :fs*10]), number_channels=1).reshape(-1) # (5000)
            if do_bandpass_filter:
                hs = denoise.heart_sound_denoise(outlier_removal(values[np.newaxis, fs*10:]), filter_lowcut, filter_highcut, fs) #  (5000)
            X[i, 0, :] = ekg
            X[i, 1, :] = hs
            
        return X
    
    warnings.filterwarnings("ignore")

    # get all filenames
    normal_filenames, abnormal_filenames = glob(normal_dir + '/*.txt'), glob(abnormal_dir + '/*.txt')
    
    # read all files and do preprocessing
    normal, abnormal = read_normal(normal_filenames), read_abnormal(abnormal_filenames)
    warnings.resetwarnings()
    
    return normal, abnormal, np.array(normal_filenames), np.array(abnormal_filenames)

def generate_survival_data(old_label_filename, new_label_filename, label_dir):
    def merge(old_df, new_df, medical_df):
        def generate_filename(df, subject_id_column, visit_code_column):
            df['filename'] = df.apply(lambda row: row[subject_id_column] + '_' + 
                                                            ('SCREENING' if row[visit_code_column] == 888 else
                                                            'V' + str(row[visit_code_column])) + 
                                                            '_Snapshot.txt', axis=1)
            df = df[~df['filename'].duplicated()] # remove duplicated entries
            return df

        # generate filename attribute
        old_df = generate_filename(old_df, 'SubjectID', 'API_Visit')

        # fix TVGH025 End_Point_visit code manually
        new_df.iloc[280, 3] = 3 

        # generate filename attribute
        new_df = generate_filename(new_df, 'Subject ID', 'End_Piont_visit')

        # there should be no duplicated entries
        assert new_df[new_df['filename'].isin(new_df[new_df['filename'].duplicated()]['filename'])].empty,\
                "There're duplicated entries in merged dataframe, please check!"

        # generate filename attribute
        medical_df = generate_filename(medical_df, 'Subject ID', 'Medical_History_visit')
        medical_df.rename(columns={
            'Visit Date': 'medical_visit_date'
        }, inplace=True)

        # merge the two dataframes by filenames
        merged_df = pd.DataFrame()

        merged_df['filename'] = old_df['filename']
        merged_df = merged_df.append(new_df[['filename']], ignore_index=True)[['filename']] # append new_df's filename to the merged_df
        merged_df = merged_df.append(medical_df[['filename']], ignore_index=True)[['filename']] # append medical_df's filename to the merged_df

        merged_df = merged_df.sort_values('filename').reset_index(drop=True) # sort by filename
        merged_df = merged_df[~merged_df['filename'].duplicated()] # remove duplicated entries

        # merge visit dates from all the dataframes
        merged_df = merged_df.merge(old_df[['filename', 'VisitDate']], left_on='filename', right_on='filename', how='left')
        merged_df = merged_df.merge(new_df[['filename', 'Visit Date']], left_on='filename', right_on='filename', how='left')
        merged_df = merged_df.merge(medical_df[['filename', 'medical_visit_date']], left_on='filename', right_on='filename', how='left')

        merged_df['measurement_date'] = merged_df['VisitDate']
        merged_df.loc[merged_df.measurement_date.isnull(), 'measurement_date'] = merged_df['Visit Date']
        merged_df.loc[merged_df.measurement_date.isnull(), 'measurement_date'] = merged_df['medical_visit_date']

        # drop used attributes
        merged_df = merged_df.drop(['VisitDate', 'Visit Date', 'medical_visit_date'], axis=1, errors='ignore')

        return merged_df

    def append_follow_up_date(merged_df, new_followup_df):
        # merge by subject ID
        merged_df['subject_id'] = merged_df.apply(lambda row: row['filename'].split('_')[0], axis=1) # TODO: FIX MMH_{} issue
        merged_df = merged_df.merge(new_followup_df[['Subject ID', 'End_Point_FollowUpDate']], left_on='subject_id', right_on='Subject ID', how='left')

        # rename and drop used columns
        merged_df = merged_df.rename(columns={'End_Point_FollowUpDate': 'follow_up_date'}, errors='ignore')
        merged_df = merged_df.drop(['Subject ID'], axis=1, errors='ignore')

        return merged_df

    def append_ADHF_dates(merged_df, new_followup_df):
        def find_adhfs(row):
            if not any(new_followup_df['Subject ID'] == row.subject_id): # cannot find subject id
                return np.nan
            adhfs = new_followup_df[new_followup_df['Subject ID'] == row.subject_id][['End_Point_ADHF_DT{:d}'.format(i) for i in range(1, 5)]].values.tolist()[0]
            adhfs = [date for date in adhfs if date == date]
            return adhfs
        merged_df['ADHF_dates'] = merged_df.apply(find_adhfs, axis=1)
        merged_df.dropna(axis=0, how='any', inplace=True)
        return merged_df

    def append_event_dates(merged_df, new_followup_df, event_name, new_event_name):
        def find_event(row):
            return new_followup_df[new_followup_df['Subject ID'] == row.subject_id]['End_Point_{}_DT'.format(event_name)].values[0]
        
        merged_df['{}_date'.format(new_event_name)] = merged_df.apply(find_event, axis=1)
        return merged_df

    def fix_followup_date(merged_df, new_event_names):
        def use_the_latest_date(row):
            dates = list()
            # get adhf dates
            for date in merged_df[merged_df.subject_id == row.subject_id].iloc[0].ADHF_dates:
                dates.append(datetime.strptime(date, '%m/%d/%Y')) # parse as a datetime object
            
            # get event dates
            for en in new_event_names:
                date = merged_df[merged_df.subject_id == row.subject_id].iloc[0]['{}_date'.format(en)]
                if date == date: # not nan
                    dates.append(datetime.strptime(date, '%m/%d/%Y')) # parse as a datetime object

            # get measurement dates
            for date in merged_df[merged_df.subject_id == row.subject_id].measurement_date:
                dates.append(datetime.strptime(date, '%Y/%m/%d'))

            # get follow_up_date
            dates.append(datetime.strptime(row.follow_up_date, '%Y/%m/%d'))
            return max(dates).strftime('%Y/%m/%d') # return the latest date

        merged_df['follow_up_date'] = merged_df.apply(use_the_latest_date, axis=1)
        return merged_df

    def generate_ADHF_data(merged_df):
        def generate_survival_data(row):
            '''
                cs: 1 - event occurred
                    0 - survived
                    -1 - pass event survived
            '''
            measurement_time = datetime.strptime(row.measurement_date, '%Y/%m/%d')
            for adhf_date in row.ADHF_dates:
                survival_time = (datetime.strptime(adhf_date, '%m/%d/%Y') - measurement_time).days
                if survival_time >= 0:
                    return pd.Series((survival_time, 1))
            
            survival_time = (datetime.strptime(row.follow_up_date, '%Y/%m/%d') - measurement_time).days
            if(len(row.ADHF_dates) > 0): # pass event
                return pd.Series((survival_time, -1))
            
            return pd.Series(( survival_time, 0))

        # generate ADHF dates
        merged_df[['ADHF_survival_time', 'ADHF_censoring_status']] = merged_df.apply(generate_survival_data, axis=1)

        return merged_df

    def generate_event_data(merged_df, new_event_name):
        def generate_survival_data(row):
            '''
                cs: 1 - event occurred
                    0 - survived
                    -1 - ignore
            '''
            measurement_time = datetime.strptime(row.measurement_date, '%Y/%m/%d')
            full_survival_time = (datetime.strptime(row.follow_up_date, '%Y/%m/%d') - measurement_time).days
            
            if row['{}_date'.format(new_event_name)] == row['{}_date'.format(new_event_name)]: # event occurred
                survival_time = (datetime.strptime(row['{}_date'.format(new_event_name)], '%m/%d/%Y') - measurement_time).days
                if survival_time >= 0:
                    return pd.Series((survival_time, 1))
                return pd.Series((full_survival_time, -1)) # pass event
            else:
                return pd.Series((full_survival_time, 0))
        
        merged_df[['{}_survival_time'.format(new_event_name), '{}_censoring_status'.format(new_event_name)]] = merged_df.apply(generate_survival_data, axis=1)
        return merged_df

    def append_info(df, sex_df, height_df):
        def interpolation(row):
            def interp(A, B, alpha, beta):
                return A + (B-A) * (alpha / (alpha-beta))
                    
            for to_fix in ['height', 'weight']:
                if row[to_fix] == row[to_fix]: # is not nan, no need to fix
                    continue
                
                # find the two closest points with the same subject id
                subject = df[(df.subject_id == row.subject_id) & 
                                (df.filename != row.filename) &
                                (~df[to_fix].isnull())]

                get_delta_time_fn = lambda md:(datetime.strptime(row.measurement_date, '%Y/%m/%d') - datetime.strptime(md, '%Y/%m/%d')).days
                delta_time = subject.measurement_date.apply(get_delta_time_fn).values
                
                # find left and right
                left, right = [np.nan, -np.Inf], [np.nan, np.Inf]
                for i, dt in enumerate(delta_time):
                    if dt <= 0 and dt > left[1]:
                        left = i, dt
                        
                    if dt > 0 and dt < right[1]:
                        right = i, dt
                        
                if left[1] == -np.Inf and right[1] == np.Inf: # cannot do interpolation
                    continue
                if left[1] == -np.Inf:
                    row[to_fix] = subject.iloc[right[0]][to_fix]
                    continue
                if right[1] == np.Inf:
                    row[to_fix] = subject.iloc[left[0]][to_fix]
                    continue
                
                closest_indices = np.abs(delta_time).argsort()[:2]
                
                if closest_indices.shape[0] == 1:
                    # append the closest point
                    row[to_fix] = subject.iloc[closest_indices][to_fix].iloc[0]
                    continue
                if closest_indices.shape[0] == 0:
                    continue
                
                A = subject.iloc[left[0]][to_fix]
                B = subject.iloc[right[0]][to_fix]
                alpha, beta = left[1], right[1]
                row[to_fix] = interp(A, B, alpha, beta)
                
            return row

        def cal_age(row):
            isnan = lambda X: X != X
            if not isnan(row.measurement_date) and not isnan(row.birthday):
                return (datetime.strptime(row.measurement_date, '%Y/%m/%d') - datetime.strptime(row.birthday, '%m/%d/%Y')).days // 365
            return np.nan

        # sex and age
        # merge by subject id
        df = df.merge(sex_df[['Subject ID', 'Study_sex', 'Study_BD']], left_on='subject_id', right_on='Subject ID', how='left')

        # rename columns
        df = df.rename(columns={
            'Study_sex': 'sex',
            'Study_BD': 'birthday'
        }).drop('Subject ID', axis=1) # and drop unnecessary columns

        # calculate age
        df.loc[321, 'birthday'] = '12/01/1928' # fix typo
        df['age'] = df.apply(cal_age, axis=1)

        # preprocessing height_df
        height_df['filename'] = height_df.apply(lambda row: row['Subject ID'] + ('_SCREENING_Snapshot.txt' if row['High_Weight_visit'] == 888 else '_V{:d}_Snapshot.txt'.format(row['High_Weight_visit'])), axis=1)

        # merge by filename
        df = df.merge(height_df[['filename', 'Height_Weight_Height', 'Height_Weight_Weight', 'Height_Weight_BMI']], left_on='filename', right_on='filename', how='left')

        # replace NIL with nan
        df.Height_Weight_BMI = df.Height_Weight_BMI.replace('NIL', np.nan)

        # rename columns
        df = df.rename(columns={
            'Height_Weight_Height': 'height',
            'Height_Weight_Weight': 'weight',
            'Height_Weight_BMI': 'BMI'
        })

        # drop dup filenames
        df.drop_duplicates(subset='filename', keep='first', inplace=True)
        df = df.apply(interpolation, axis=1) # do height & weight interpolation

        # re-calculate BMI
        df['BMI'] = df.weight / (df.height**2 * 0.0001)

        # fill nan with mean / medium / mode
        df['sex'] = df[['sex']].fillna(df.sex.mode().values[0])
        df[['age', 'height', 'weight', 'BMI']] = df[['age', 'height', 'weight', 'BMI']].fillna(df[['age', 'height', 'weight', 'BMI']].mean())

        return df

    # read all files
    old_df, new_df = pd.read_excel(old_label_filename), pd.read_excel(new_label_filename)
    new_followup_df = pd.read_excel(new_label_filename, sheet_name=3) # the third sheet which has follow up data
    medical_df = pd.read_excel(os.path.join(label_dir, 'HF03. Medical History.xlsx'), skiprows=[0, 1, 3]) # medical history

    sex_df = pd.read_excel(os.path.join(label_dir, 'HF01. Demongraphy.xlsx'), skiprows=[0, 1, 3]) # sex and age
    height_df = pd.read_excel(os.path.join(label_dir, 'HF04. Height and Weight.xlsx'), skiprows=[0, 1, 3]) # height, weight and BMI

    # fix FEMH012's CV_death possible mistake
    new_followup_df.loc[new_followup_df['Subject ID'] == 'FEMH012', 'End_Point_CV_death_DT'] = '10/01/2019'

    # merge them by filename, get measurement_date
    merged_df = merge(old_df, new_df, medical_df)

    merged_df = append_follow_up_date(merged_df, new_followup_df)
    # merged_df = append_ADHF(merged_df, new_followup_df)
    # find all event dates
    merged_df = append_ADHF_dates(merged_df, new_followup_df)
    for event_name, new_event_name in [['CV_death', 'CVD'], ['all_cause_death', 'Mortality']]:# , ['nonfetal_MI', 'MI']]:
        merged_df = append_event_dates(merged_df, new_followup_df, event_name, new_event_name)

    # fix follow up date
    merged_df = fix_followup_date(merged_df, ['CVD', 'Mortality'])

    # generate survival datas 
    merged_df = generate_ADHF_data(merged_df)
    for en in ['CVD', 'Mortality']:
        merged_df = generate_event_data(merged_df, en)

    # append sex, age, height, weight, BMI
    merged_df = append_info(merged_df, sex_df, height_df)

    return merged_df

if __name__ == '__main__':
    # parse config
    NORMAL_DIR, ABNORMAL_DIR = config['Audicor_10s']['normal_dir'], config['Audicor_10s']['abnormal_dir']
    OUTPUT_DIR = config['Audicor_10s']['output_dir']
    LABEL_DIR = config['Audicor_10s']['label_dir']
    OLD_LABEL_FILENAME, NEW_LABEL_FILENAME = config['Audicor_10s']['label_filenames'].split(', ')
    DO_BANDPASS_FILTER = config['Audicor_10s'].getboolean('do_bandpass_filter')
    FILTER_LOWCUT, FILTER_HIGHCUT = int(config['Audicor_10s']['bandpass_filter_lowcut']), int(config['Audicor_10s']['bandpass_filter_highcut'])

    '''
    normal_X, abnormal_X, normal_filenames, abnormal_filenames = read_data(NORMAL_DIR, 
                                                                    ABNORMAL_DIR, 
                                                                    DO_BANDPASS_FILTER, 
                                                                    FILTER_LOWCUT, FILTER_HIGHCUT, 500)

    # save data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'normal_X.npy'), normal_X)
    np.save(os.path.join(OUTPUT_DIR, 'normal_filenames.npy'), normal_filenames)

    np.save(os.path.join(OUTPUT_DIR, 'abnormal_X.npy'), abnormal_X)
    np.save(os.path.join(OUTPUT_DIR, 'abnormal_filenames.npy'), abnormal_filenames)
    '''

    abnormal_event_df = generate_survival_data(OLD_LABEL_FILENAME, NEW_LABEL_FILENAME, LABEL_DIR)
    abnormal_event_df.to_csv(os.path.join(OUTPUT_DIR, 'abnormal_event.csv'), index=False)