# Managing Anomalies in Energy Time Series for Automated Forecasting

This repository contains the Python implementation of experiments to manage anomalies in energy time series presented in the following paper:
>M. Turowski, O. Neumann, L. Mannsperger, K. Kraus, K. Layer, R. Mikut, and V. Hagenmeyer, 2024, "Managing Anomalies in Energy Time Series for Automated Forecasting," in Energy Informatics: EI.A 2023, Ed. by B. N. Jørgensen, L. C. P. da Silva, and Z. Ma, Springer Nature Switzerland, pp. 3–29. doi: [10.1007/978-3-031-48649-4_1](https://doi.org/10.1007/978-3-031-48649-4_1).


## Installation

To install this project, perform the following steps:
1. Clone the project
2. Create an conda virtual environment using `conda create -n managing_anomalies_env python=3.9`
3. Activate conda environment via `conda activate managing_anomalies_env`
4. Install Tensorflow, Torch, and Prophet using conda via:
    - `conda install -c conda-forge tensorflow==2.8.0`
    - `conda install -c conda-forge libgcc==5.2.0`
    - `conda install -c pytorch pytorch==1.11.0`
    - `conda install -c conda-forge prophet==1.0.1`
5. Install other python dependencies using `pip install -r requirements.txt` (possibly install pip before using `conda install pip`)
6. Install correct version of pywatts-pipeline:
    - `pip uninstall pywatts-pipeline`
    - `pip install git+https://github.com/KIT-IAI/pywatts-pipeline@fb7d0096343f6283e58466b2f30dee3137e5b771`


## How to run

All experiments can be run using `python run.py`. If a specific experiment should be run, use `python pipeline.py`.


### Single experiment

To run a single experiment, use the following command:

`python Name_of_the_run Path_to_ground_truth_file "Name of timestamp column" "Name of column with value" Path_to_anomalies_file Path_to_detection_file`

Example:

`python pipeline.py Experiment_Name data/in_train_ID200.csv index 0 data/out_train_ID200_20_20_20_20_unusual_behaviour.csv data/unusual/num_anomalies_20/LOF/cvae_sup-False_FunctionModule.csv`

You can add further arguments to the command as defined in the `pipeline.py`. To get an overview of the available arguments, use `python pipeline.py --help`.


### All experiments

To run all experiments in parallel, use the following python command:

`python run.py`

 

## Pipeline details

### General structure

The `pipeline.py` parses the passed arguments of a command. Based on the arguments, the `pipeline.py` calls the relevant steps. The pipeline comprises the steps compensation and forecast (see folder /steps). In each step, a pipeline is executed that can make use of methods implemented in the corresponding folder (see folder /pipelines/) and listed in the corresponding `__init__.py` file. Afterward, in each step, a hook is executed (see folder /hooks) if the parameter `--hooks` is used. The forecasting step additionally executes an evaluation pipeline that uses the metrics defined in the folder /pipelines/evaluation.

### Used Time series

Up to five different time series are used in the pipeline:

| Time series  | Description                                                                            |  
| -----------  |----------------------------------------------------------------------------------------|
| y            | Original input time series                                                             |
| y_hat        | Input time series with inserted anomalies                                              |
| y_hat_comp   | Input time series with inserted anomalies that are compensated in the compensation step |
| anomalies    | Labels of inserted anomalies                                 |
| anomalies_hat| Labels of detected anomalies                                 |


## Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and the Helmholtz Association under the Program “Energy System Design”.


## License
This code is licensed under the [MIT License](LICENSE).
