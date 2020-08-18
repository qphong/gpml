# Gaussian Process Meta-Learning for Few-shot Regression
The source code for the paper: Gaussian Process Meta-Learning for Few-shot Regression


## Getting Started

### Prerequisites
```
tqdm: 4.36.1
numpy: 1.16.4
scipy: 1.2.1
tensorflow: 1.14.0
tensorflow-probability: 0.7.0
pandas: 0.24.2
seaborn: 0.9.0
matplotlib: 3.1.1
```

### Dataset
* To run the real-world experiment, the dataset is downloaded from: <http://db.csail.mit.edu/labdata/labdata.html>, including data.txt (sensor readings by minutes) and mote_loc.txt (locations of sensors)

* Pre-processing:
    * Moving data.txt and mote_loc.txt to folder: sensor_data.
    * Removing rows in data.txt that do not have enough sensor readings for all types.
    * In folder sensor_data: Creating the timestamp attribute and removing timestamps that have less than 35 readings by running generate_sufficient_sensor_data.py to create file sufficient_sensor_data.csv (or un-compressing the file sufficient_sensor_data.csv.tar.gz)

***

## Running The Tests
To run synthetic functions:
* cosine: cosine functions with varying amplitude and phase.
* cosineline: mixture of cosine and linear functions
* noisysine: a sine function with noisy observation and varying amplitude, frequency, and phase.

Available scripts should be ran with correct parameters
* For GPML method:
```
./script_gml.sh
```
* For MAML method:
```
./script_maml.sh
```

* Light sensor experiment: requiring the downloading and processing of dataset above.

***

## Visualizing the results

### Visualizing predicted and inducing tasks
* To visualize predicted (and/or inducing) tasks trained with GPML,
```
python plot_gpml.py
```
where task_name, n_inducing_task, training_task_batchsize, training_task_datasize, n_step, use_samples, is_pivot_X_adaptive_in_Kuu, n_pivot_X should be set manually in plot_gpml.py

* To visualize predicted (and/or inducing) tasks trained with MAML,
```
python plot_maml.py
```
where task_name, training_task_batchsize, training_task_datasize, n_step should be set manually in plot_maml.py

* To visualize the distance between inducing tasks,
```
python plot_distance_to_inducing_tasks.py
```
where parameters are set manually in code.

### MSE boxplots
1. Generate the MSE result of the optimized meta model
```
python get_mse_loss.py
```
where parameters are set manually in code.

2. Plot the MSE boxplots
```
python plot_mse_boxplot.py
```
where parameters are set manually in code.

