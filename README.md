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
        |  EKG  |  HS   | Accuracy                 | Precision                | Recall                   | F1                       | ROC AUC                  | Sweep ID   |
        | :---: | :---: | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ---------- |
        |   V   |       | 0.946 / 0.9526 / 0.942   | 0.9618 / 0.9603 / 0.9538 | 0.9517 / 0.9643 / 0.9538 | 0.9567 / 0.9623 / 0.9538 | 0.9517 / 0.9642 / 0.9538 | `hgvzjjcn` |
        |       |   V   | 0.9265 / 0.9307 / 0.9298 | 0.9062 / 0.8986 / 0.8985 | 0.9805 / 0.9986 / 0.9972 | 0.9419 / 0.946 / 0.9453  | 0.9805 / 0.9986 / 0.9972 | `ih8g4x1v` |
        |   V   |   V   | 0.9696 / 0.9704 / 0.9738 | 0.983  / 0.9724 / 0.9738 | 0.9666 / 0.9791 / 0.9833 | 0.9748 / 0.9757 / 0.9785 | 0.9666 / 0.9791 / 0.9833 | `e7dyx0xq` |

* Hazard prediction - PureBigExam - Weighted Concordance Index of ADHF / Mortality / MI / CVDeath
    * With normal subjects in both training and testing sets
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID   | Note |
        | :---: | :---: | :--------------- | :--------: | :--------------: | :--------------: | ---------- | ---- |
        |   V   |       | Cox              |   0.7063   |      0.7597      |      0.7716      | `n16jsktd` |      |
        |   V   |       | Weibull AFT      |   0.7728   |      0.7661      |      0.7729      | `ddtiuvw6` |      |
        |   V   |       | Log-logistic AFT |   0.7522   |      0.7730      |      0.7833      | `pr4gp97a` |      |
        |       |   V   | Cox              |   0.7010   |      0.7047      |      0.7377      | `d3aitvdx` |      |
        |       |   V   | Weibull AFT      |   0.7299   |      0.7455      |      0.7453      | `gkcpwxkj` |      |
        |       |   V   | Log-logistic AFT |   0.6942   |      0.7374      |      0.7419      | `50ioyk9h` |      |
        |   V   |   V   | Cox              |   0.7829   |      0.7848      |      0.7925      | `jkjs2l3l` |      |
        |   V   |   V   | Weibull AFT      |   0.7265   |      0.7371      |      0.7833      | `w98wwvtq` |      |
        |   V   |   V   | Log-logistic AFT |   0.7771   |      0.7709      |      0.7716      | `1xhq57sl` |      |

    * Without normal subjects
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID   | Note                          |
        | :---: | :---: | :--------------- | :--------: | :--------------: | :--------------: | ---------- | ----------------------------- |
        |   V   |       | Cox              |            |                  |                  |            |                               |
        |   V   |       | Weibull AFT      |            |                  |                  |            |                               |
        |   V   |       | Log-logistic AFT |            |                  |                  |            |                               |
        |       |   V   | Cox              |            |                  |                  |            |                               |
        |       |   V   | Weibull AFT      |            |                  |                  |            |                               |
        |       |   V   | Log-logistic AFT |            |                  |                  |            |                               |
        |   V   |   V   | Cox              |   0.5726   |      0.6159      |      0.6114      | `nelnru6g` | sex, age, height, weight, BMI |
        |   V   |   V   | Weibull AFT      |   0.4553   |      0.5147      |      0.6108      | `6ep13r9l` | sex, age, height, weight, BMI |
        |   V   |   V   | Log-logistic AFT |   0.6108   |      0.6268      |      0.6174      | `rjmo63l5` | sex, age, height, weight, BMI |


* Hazard prediction - Hybrid - Weighted Concordance Index of ADHF / Mortality
    * With normal subjects in both training and testing sets
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID   | Note |
        | :---: | :---: | :--------------- | :--------: | :--------------: | :--------------: | ---------- | ---- |
        |   V   |       | Cox              |   0.6770   |      0.6988      |      0.6857      | `uunm1m1t` |      |
        |   V   |       | Weibull AFT      |   0.6274   |      0.6813      |      0.6919      | `j6bb43dc` |      |
        |   V   |       | Log-logistic AFT |   0.7114   |      0.7072      |      0.7050      | `vmagsbod` |      |
        |       |   V   | Cox              |   0.7545   |      0.7654      |      0.7696      | `9zimatkj` |      |
        |       |   V   | Weibull AFT      |   0.7422   |      0.7575      |      0.7717      | `x4kxis1t` |      |
        |       |   V   | Log-logistic AFT |   0.7560   |      0.7600      |      0.7718      | `chuzr55g` |      |
        |   V   |   V   | Cox              |   0.7607   |      0.7605      |      0.7668      | `y8nbsgof` |      |
        |   V   |   V   | Weibull AFT      |   0.7020   |      0.6912      |      0.7064      | `qj9l0u63` |      |
        |   V   |   V   | Log-logistic AFT |   0.7873   |      0.7777      |      0.7896      | `3g9aqusu` |      |
        
    * Without normal subjects
        |  EKG  |  HS   | Survival Model   | Best Model | 3-Model Ensemble | 5-Model Ensemble | Sweep ID   | Note          |
        | :---: | :---: | :--------------- | :--------: | :--------------: | :--------------: | ---------- | ------------- |
        |   V   |       | Cox              |            |                  |                  |            |               |
        |   V   |       | Weibull AFT      |            |                  |                  |            |               |
        |   V   |       | Log-logistic AFT |            |                  |                  |            |               |
        |       |   V   | Cox              |            |                  |                  |            |               |
        |       |   V   | Weibull AFT      |            |                  |                  |            |               |
        |       |   V   | Log-logistic AFT |            |                  |                  |            |               |
        |   V   |   V   | Cox              |   0.6427   |      0.6362      |      0.6324      | `m6gmavxd` | sex, age, BMI |
        |   V   |   V   | Weibull AFT      |   0.6206   |      0.6235      |      0.6507      | `phltanpf` | sex, age, BMI |
        |   V   |   V   | Log-logistic AFT |   0.6344   |      0.6351      |      0.6302      | `ge44yydf` | sex, age, BMI |