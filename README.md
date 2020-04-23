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
| # of Abnormal Subjects    	| 209                              	| ~170             	|
| # of Abnormal Signals     	| 852                             	| ~776            	|
| # of EKG Channels         	| 8                                	| 1               	|
| # of Heart Sound Channels 	| 2                                	| 1               	|
| Sampling Rate             	| 1000                             	| 500             	|
| Signal Length             	| 10 sec                           	| 10 sec          	|
| Events                    	| ADHF, MI, Stroke, CVD, Mortality 	| ADHF, Mortality 	|
| Longest Follow-up Days    	| 2872 days                        	| 418 days        	|
| Longest Event Days    	    |                                  	|                 	|

## Results
* Notation
    * `B`: big_exam
    * `A`: audicor_10s
* Abnormal detection
    * Best Model / 3-Model Ensemble / 5-Model Ensemble - `sweep_id = toosyou/ekg-abnormal_detection/078xfb5r`
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision         | Recall          | F1              |
        |--------------------|-----------------|---------------|-----------------------|-------------------|-----------------|-----------------|
        | 0.49 * B           | 0.21 * B        | 0.3 * B       |                       |                   |                 |                 |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       |                       |                   |                 |                 |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                       |                   |                 |                 |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.997 / 0.994 / 0.994 | 1.0 / 0.99 / 0.99 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |

    * Only EKG - Best Model / 3-Model Ensemble / 5-Model Ensemble - `sweep_id = toosyou/ekg-abnormal_detection/9kdjnovy`
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision          | Recall          | F1                 |
        |--------------------|-----------------|---------------|-----------------------|--------------------|-----------------|--------------------|
        | 0.49 * B           | 0.21 * B        | 0.3 * B       |                       |                    |                 |                    |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       |                       |                    |                 |                    |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                       |                    |                 |                    |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.982 / 0.979 / 0.982 | 0.97 / 0.97 / 0.97 | 1.0 / 1.0 / 1.0 | 0.99 / 0.99 / 0.99 |

    * Only Heart Sound - Best Model / 3-Model Ensemble / 5-Model Ensemble - `sweep_id = toosyou/ekg-abnormal_detection/ccbq03an`
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision          | Recall          | F1                 |
        |--------------------|-----------------|---------------|-----------------------|--------------------|-----------------|--------------------|
        | 0.49 * B           | 0.21 * B        | 0.3 * B       |                       |                    |                 |                    |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       |                       |                    |                 |                    |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                       |                    |                 |                    |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.982 / 0.979 / 0.982 | 0.97 / 0.97 / 0.97 | 1.0 / 1.0 / 1.0 | 0.99 / 0.99 / 0.99 |

* Hazard prediction
    * Both EKG and Heart Sound
        * Concordance Index - ADHF / Mortality - `sweep_id = toosyou/ekg-hazard_prediction/761vrbqj`
            | Training data      | Validation Data | Testing Data  | Best Model      | 3-Model Ensemble | 5-Model Ensemble |
            |--------------------|-----------------|---------------|-----------------|------------------|------------------|
            | 0.49 * B           | 0.21 * B        | 0.3 * B       |                 |                  |                  |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       |                 |                  |                  |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                 |                  |                  |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.7542 / 0.9004 | 0.7533 / 0.8699  | 0.7715 / 0.8804  |
    * Only EKG
        * Concordance Index - ADHF / Mortality - `sweep_id = toosyou/ekg-hazard_prediction/ic4ttlqf`
            | Training data      | Validation Data | Testing Data  | Best Model      | 3-Model Ensemble | 5-Model Ensemble |
            |--------------------|-----------------|---------------|-----------------|------------------|------------------|
            | 0.49 * B           | 0.21 * B        | 0.3 * B       |                 |                  |                  |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       |                 |                  |                  |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                 |                  |                  |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.5995 / 0.6342 | 0.6539 / 0.6143  | 0.6536 / 0.6213  |
    * Only Heart Sound
        * Concordance Index - ADHF / Mortality - `sweep_id = toosyou/ekg-hazard_prediction/4efr1zi8`
            | Training data      | Validation Data | Testing Data  | Best Model      | 3-Model Ensemble | 5-Model Ensemble |
            |--------------------|-----------------|---------------|-----------------|------------------|------------------|
            | 0.49 * B           | 0.21 * B        | 0.3 * B       |                 |                  |                  |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       |                 |                  |                  |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) |                 |                  |                  |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.7267 / 0.8042 | 0.7386 / 0.8945  | 0.7545 / 0.9355  |
