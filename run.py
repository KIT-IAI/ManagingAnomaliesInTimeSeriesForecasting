import os
import multiprocessing


# Number of parallel processes. Some methods use more CPU than others, so this 
# should be figured out experimentally.
number_of_parallel_processes = 48

## experiments to be performed
experiments = [
    "baseline_forecast",
    "raw_forecast",
    "forecast_with_detected_anomalies",
    "forecast_with_compensated_anomalies"
]


all_processes = []
for seed in range(1, 2):
    # Add commands to process list for all experiments
    for e in experiments:
        # UCI technical
        # - unsupervised
        if "forecast_with_compensated_anomalies" in e:
            all_processes.append("python pipeline.py UCI-Technical-Unsup data/in_train_ID200.csv \"index\" \"0\" data/out_train_ID200_20_20_20_20_small.csv \"data/technical/num_anomalies_20/VAE/cinn_sup:False_FunctionModule.csv\" --insertion_method identity --compensation_method my_prophet --experiment " + str(e) + f" --seed {seed} --detection_type unsupervised --logging svr/run{seed}/results6_" + str(e) + ".csv")
        else:
            all_processes.append("python pipeline.py UCI-Technical-Unsup data/in_train_ID200.csv \"index\" \"0\" data/out_train_ID200_20_20_20_20_small.csv \"data/technical/num_anomalies_20/VAE/cinn_sup:False_FunctionModule.csv\" --insertion_method identity --compensation_method identity --experiment " + str(e) + f" --seed {seed} --detection_type unsupervised --logging svr/run{seed}/results6_" + str(e) + ".csv")

        # UCI unusual
        # - unsupervised
        if "forecast_with_compensated_anomalies" in e:
            all_processes.append("python pipeline.py UCI-Unusual-Unsup data/in_train_ID200.csv \"index\" \"0\" data/out_train_ID200_20_20_20_20_unusual_behaviour.csv \"data/unusual/num_anomalies_20/LOF/cvae_sup:False_FunctionModule.csv\" --insertion_method identity --compensation_method my_prophet --experiment " + str(e) + f" --seed {seed} --detection_type unsupervised --logging svr/run{seed}/results16_" + str(e) + ".csv")
        else:
            all_processes.append("python pipeline.py UCI-Unusual-Unsup data/in_train_ID200.csv \"index\" \"0\" data/out_train_ID200_20_20_20_20_unusual_behaviour.csv \"data/unusual/num_anomalies_20/LOF/cvae_sup:False_FunctionModule.csv\" --insertion_method identity --compensation_method identity --experiment " + str(e) + f" --seed {seed} --detection_type unsupervised --logging svr/run{seed}/results16_" + str(e) + ".csv")

def execute(process):
    os.system(f'{process}')

# All commands are now in one single list and are then executed.
process_pool = multiprocessing.Pool(processes = number_of_parallel_processes)
process_pool.map(execute, all_processes)
