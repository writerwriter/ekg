# EKG

## Before Experiments
1. Adjust `config.cfg`.
1. Run the following commands to generate data to `data` directory
    * `python3 scripts/data_preprocessing/big_exam.py`
    * `python3 scripts/data_preprocessing/audicor_10s.py`

## Generated Data
1. `big_exam`
    * `abnormal_X.npy`
        * shape: `[n_instances, n_channels, n_samples]`
    * `normal_X.npy`
        * shape: `[n_instances, n_channels, n_samples]`
    * `abnormal_event.csv`
        * events
            * ADHF
            * MI
            * Stroke
            * CVD
            * Mortality
        * attributes
            * filename
            * subject_id
            * ADHF_censoring_status
            * ADHF_survival_time
            * MI_censoring_status
            * MI_survival_time
            * Stroke_censoring_status
            * Stroke_survival_time
            * CVD_censoring_status
            * CVD_survival_time
            * Mortality_censoring_status
            * Mortality_survival_time
2. `audicor_10s`
    * `abnormal_X.npy`
        * shape: `[n_instances, n_channels, n_samples]`
    * `normal_X.npy`
        * shape: `[n_instances, n_channels, n_samples]`
    * `abnormal_filenames.npy`
        * shape: `[n_instances]`
    * `normal_filenames.npy`
        * shape: `[n_instances]`
    * `abnormal_event.csv`
        * events
            * ADHF
            * CVD
            * Mortality
        * attributes
            * filename
            * measurement_date
            * subject_id
            * follow_up_date
            * ADHF_dates
            * ADHF_survival_time
            * ADHF_censoring_status
            * CVD_date
            * CVD_survival_time
            * CVD_censoring_status
            * Mortality_date
            * Mortality_survival_time
            * Mortality_censoring_status

## Data Statistics
| Data                      	| big_exam                         	| audicor_10s     	|
|---------------------------	|----------------------------------	|-----------------	|
| # of Normal Subjects      	| 601                              	| 340              	|
| # of Normal Signals       	| 601                              	| 340              	|
| # of Abnormal Subjects    	| 226                              	| 211             	|
| # of Abnormal Signals     	| 795                             	| 776            	|
| # of EKG Channels         	| 8                                	| 1               	|
| # of Heart Sound Channels 	| 2                                	| 1               	|
| Sampling Rate             	| 1000 Hz                          	| 500 Hz          	|
| Signal Length             	| 10 sec                           	| 10 sec          	|
| Events                    	| ADHF, MI, Stroke, CVD, Mortality 	| ADHF, Mortality 	|
| Longest Follow-up Time    	| 2872 days                        	| 418 days        	|
| Longest Event Time    	    | 2048 days                         | 322 days         	|

## Results
* Notation
    * `B`: big_exam
    * `A`: audicor_10s
    * Data setting      
        | Name        | Training data  | Validation Data | Testing Data  |
        | ----------- | -------------- | --------------- | ------------- |
        | PureBigExam | 0.49 * B       | 0.21 * B        | 0.3 * B       |
        | Hybrid      | 0.49 * (B + A) | 0.21 * (B + A)  | 0.3 * (B + A) |

* Abnormal detection - Hybrid
    * Best Model / 3-Model Ensemble / 5-Model Ensemble
        |  EKG  |  HS   | Accuracy | F1  | ROC AUC | Sweep ID |
        | :---: | :---: | -------- | --- | ------- | -------- |
        |   V   |       |          |     |         |          |
        |       |   V   |          |     |         |          |
        |   V   |   V   |          |     |         |          |

* Hazard prediction - PureBigExam - Concordance Index: ADHF / Mortality / MI / CVDeath
    * With normal subjects in both training and testing sets
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID | Note |
        | :---: | :---: | :--------------- | ---------- | ---------------- | ---------------- | -------- | ---- |
        |   V   |       | Cox              |            |                  |                  |          |      |
        |   V   |       | Weibull AFT      |            |                  |                  |          |      |
        |   V   |       | Log-logistic AFT |            |                  |                  |          |      |
        |       |   V   | Cox              |            |                  |                  |          |      |
        |       |   V   | Weibull AFT      |            |                  |                  |          |      |
        |       |   V   | Log-logistic AFT |            |                  |                  |          |      |
        |   V   |   V   | Cox              |            |                  |                  |          |      |
        |   V   |   V   | Weibull AFT      |            |                  |                  |          |      |
        |   V   |   V   | Log-logistic AFT |            |                  |                  |          |      |

    * Without normal subjects
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID | Note |
        | :---: | :---: | :--------------- | ---------- | ---------------- | ---------------- | -------- | ---- |
        |   V   |       | Cox              |            |                  |                  |          |      |
        |   V   |       | Weibull AFT      |            |                  |                  |          |      |
        |   V   |       | Log-logistic AFT |            |                  |                  |          |      |
        |       |   V   | Cox              |            |                  |                  |          |      |
        |       |   V   | Weibull AFT      |            |                  |                  |          |      |
        |       |   V   | Log-logistic AFT |            |                  |                  |          |      |
        |   V   |   V   | Cox              |            |                  |                  |          |      |
        |   V   |   V   | Weibull AFT      |            |                  |                  |          |      |
        |   V   |   V   | Log-logistic AFT |            |                  |                  |          |      |


* Hazard prediction - Hybrid - Concordance Index: ADHF / Mortality
    * With normal subjects in both training and testing sets
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID | Note |
        | :---: | :---: | :--------------- | ---------- | ---------------- | ---------------- | -------- | ---- |
        |   V   |       | Cox              |            |                  |                  |          |      |
        |   V   |       | Weibull AFT      |            |                  |                  |          |      |
        |   V   |       | Log-logistic AFT |            |                  |                  |          |      |
        |       |   V   | Cox              |            |                  |                  |          |      |
        |       |   V   | Weibull AFT      |            |                  |                  |          |      |
        |       |   V   | Log-logistic AFT |            |                  |                  |          |      |
        |   V   |   V   | Cox              |            |                  |                  |          |      |
        |   V   |   V   | Weibull AFT      |            |                  |                  |          |      |
        |   V   |   V   | Log-logistic AFT |            |                  |                  |          |      |
        
    * Without normal subjects
        |  EKG  |  HS   | Survival Model   | Best Model      | 3-Model Ensemble | 5-Model Ensemble | Sweep ID   | Note |
        | :---: | :---: | :--------------- | --------------- | ---------------- | ---------------- | ---------- | ---- |
        |   V   |       | Cox              |                 |                  |                  |            |      |
        |   V   |       | Weibull AFT      |                 |                  |                  |            |      |
        |   V   |       | Log-logistic AFT |                 |                  |                  |            |      |
        |       |   V   | Cox              |                 |                  |                  |            |      |
        |       |   V   | Weibull AFT      |                 |                  |                  |            |      |
        |       |   V   | Log-logistic AFT |                 |                  |                  |            |      |
        |   V   |   V   | Cox              | 0.5956 / 0.6201 | 0.5936 / 0.6359  | 0.5948 / 0.6493  | `ldui46cc` |      |
        |   V   |   V   | Weibull AFT      | 0.5869 / 0.6537 | 0.6111 / 0.6642  | 0.6148 / 0.6594  | `d345t3sz` |      |
        |   V   |   V   | Log-logistic AFT | 0.6125 / 0.6594 | 0.6198 / 0.6596  | 0.6140 / 0.6574  | `g1aqnc4r` |      |