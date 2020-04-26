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
| # of Abnormal Subjects    	| 209                              	| 211             	|
| # of Abnormal Signals     	| 852                             	| 776            	|
| # of EKG Channels         	| 8                                	| 1               	|
| # of Heart Sound Channels 	| 2                                	| 1               	|
| Sampling Rate             	| 1000 Hz                          	| 500 Hz          	|
| Signal Length             	| 10 sec                           	| 10 sec          	|
| Events                    	| ADHF, MI, Stroke, CVD, Mortality 	| ADHF, Mortality 	|
| Longest Follow-up Time    	| 2872 days                        	| 418 days        	|
| Longest Event Time    	    |                                  	| 322 days         	|

## Results
* Notation
    * `B`: big_exam
    * `A`: audicor_10s
* Abnormal detection
    * Best Model / 3-Model Ensemble / 5-Model Ensemble
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision          | Recall             | F1                 | Sweep ID   |
        | ------------------ | --------------- | ------------- | --------------------- | ------------------ | ------------------ | ------------------ | ---------- |
        | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.950 / 0.962 / 0.950 | 0.96 / 0.96 / 0.95 | 0.95 / 0.98 / 0.97 | 0.96 / 0.97 / 0.98 | `l0bmvfd5` |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       |                       |                    |                    |                    |            |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.949 / 0.967 / 0.960 | 0.95 / 0.97 / 0.95 | 0.97 / 0.98 / 0.98 | 0.96 / 0.97 / 0.97 | `vct4ohx8` |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.997 / 0.994 / 0.994 | 1.0 / 0.99 / 0.99  | 1.0 / 1.0 / 1.0    | 1.0 / 1.0 / 1.0    | `078xfb5r` |
    * Only EKG - Best Model / 3-Model Ensemble / 5-Model Ensemble
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision          | Recall             | F1                 | Sweep ID   |
        | ------------------ | --------------- | ------------- | --------------------- | ------------------ | ------------------ | ------------------ | ---------- |
        | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.948 / 0.950 / 0.957 | 0.96 / 0.96 / 0.96 | 0.95 / 0.95 / 0.96 | 0.95 / 0.96 / 0.96 | `xkggzs8k` |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       | 0.943 / 0.943 / 0.943 | 0.96 / 0.94 / 0.94 | 0.96 / 0.98 / 0.98 | 0.96 / 0.96 / 0.96 | `8wuvz2qz` |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.911 / 0.913 / 0.915 | 0.90 / 0.91 / 0.91 | 0.96 / 0.96 / 0.96 | 0.93 / 0.93 / 0.93 | `ngc9trkg` |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.982 / 0.979 / 0.982 | 0.97 / 0.97 / 0.97 | 1.0 / 1.0 / 1.0    | 0.99 / 0.99 / 0.99 | `9kdjnovy` |

    * Only Heart Sound - Best Model / 3-Model Ensemble / 5-Model Ensemble
        | Training data      | Validation Data | Testing Data  | Accuracy              | Precision          | Recall             | F1                 | Sweep ID   |
        | ------------------ | --------------- | ------------- | --------------------- | ------------------ | ------------------ | ------------------ | ---------- |
        | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.865 / 0.891 / 0.889 | 0.86 / 0.86 / 0.86 | 0.92 / 0.98 / 0.97 | 0.89 / 0.91 / 0.91 | `dt2bivxc` |
        | 0.49 * A           | 0.21 * A        | 0.3 * A       | 0.698 / 0.698 / 0.698 | 0.70 / 0.7 / 0.7   | 1.0 / 1.0 / 1.0    | 0.82 / 0.82 / 0.82 | `d3bbp2sh` |
        | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.919 / 0.920 / 0.923 | 0.89 / 0.89 / 0.89 | 0.99 / 0.99 / 0.99 | 0.94 / 0.94 / 0.94 | `zx53fukt` |
        | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.982 / 0.979 / 0.982 | 0.97 / 0.97 / 0.97 | 1.0 / 1.0 / 1.0    | 0.99 / 0.99 / 0.99 | `ccbq03an` |

* Hazard prediction
    * Both EKG and Heart Sound
        * Concordance Index - ADHF / Mortality / (MI / Stroke / CVD)
            | Training data      | Validation Data | Testing Data  | Best Model                                 | 3-Model Ensemble                           | 5-Model Ensemble                           | Sweep ID   |
            | ------------------ | --------------- | ------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ---------- |
            | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.7794 / 0.7604 / 0.6648 / 0.7103 / 0.7481 | 0.7934 / 0.7756 / 0.7086 / 0.6783 / 0.7742 | 0.7936 / 0.7798 / 0.7302 / 0.6990 / 0.7822 | `8zm1dn71` |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       | 0.7598 / 0.8687                            | 0.7593 / 0.9414                            | 0.7709 / 0.8687                            | `4y4tn330` |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.7204 / 0.7409                            | 0.7172 / 0.7384                            | 0.7209 / 0.7445                            | `9kguu504` |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.7542 / 0.9004                            | 0.7533 / 0.8699                            | 0.7715 / 0.8804                            | `761vrbqj` |
    * Only EKG
        * Concordance Index - ADHF / Mortality / (MI / Stroke / CVD)
            | Training data      | Validation Data | Testing Data  | Best Model                                 | 3-Model Ensemble                           | 5-Model Ensemble                           | Sweep ID   |
            | ------------------ | --------------- | ------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ---------- |
            | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.7798 / 0.8027 / 0.7516 / 0.5389 / 0.8026 | 0.8073 / 0.8149 / 0.7960 / 0.7188 / 0.8271 | 0.8136 / 0.8138 / 0.7986 / 0.7163 / 0.8269 | `rs3j8ag3` |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       | 0.7049 / 0.5463                            | 0.7097 / 0.6120                            | 0.7182 / 0.6166                            | `i1ezjbxu` |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.7365 / 0.6743                            | 0.7039 / 0.6459                            | 0.7074 / 0.6473                            | `v3lhpi8b` |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.5995 / 0.6342                            | 0.6539 / 0.6143                            | 0.6536 / 0.6213                            | `ic4ttlqf` |
    * Only Heart Sound
        * Concordance Index - ADHF / Mortality / (MI / Stroke / CVD)
            | Training data      | Validation Data | Testing Data  | Best Model                                 | 3-Model Ensemble                           | 5-Model Ensemble                           | Sweep ID   |
            | ------------------ | --------------- | ------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ---------- |
            | 0.49 * B           | 0.21 * B        | 0.3 * B       | 0.7691 / 0.8022 / 0.7314 / 0.8832 / 0.7956 | 0.7521 / 0.8030 / 0.7363 / 0.8647 / 0.7988 | 0.7502 / 0.7942 / 0.7142 / 0.8597 / 0.7884 | `ecqefeg5` |
            | 0.49 * A           | 0.21 * A        | 0.3 * A       | 0.8015 / 0.7890                            | 0.7627 / 0.9261                            | 0.7420 / 0.9097                            | `vrq7unie` |
            | 0.49 * (B + A)     | 0.21 * (B + A)  | 0.3 * (B + A) | 0.7299 / 0.7647                            | 0.7336 / 0.7865                            | 0.7326 / 0.7925                            | `4xtp5skh` |
            | 1.0 * B + 0.49 * A | 0.21 * A        | 0.3 * A       | 0.7267 / 0.8042                            | 0.7386 / 0.8945                            | 0.7545 / 0.9355                            | `4efr1zi8` |